import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


def _install_stub_modules():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    jieba = types.ModuleType("jieba")
    jieba.cut_for_search = lambda text: str(text).split()
    jieba.cut = lambda text: str(text).split()
    sys.modules["jieba"] = jieba

    numpy = types.ModuleType("numpy")
    numpy.float32 = float
    numpy.int64 = int
    numpy.ndarray = list
    numpy.array = lambda value, dtype=None: list(value) if isinstance(value, (list, tuple)) else [value]
    numpy.stack = lambda values: list(values)
    numpy.vstack = lambda values: list(values)
    numpy.concatenate = lambda values: [item for seq in values for item in seq]
    numpy.argsort = lambda values: sorted(range(len(values)), key=lambda i: values[i])
    numpy.load = lambda *args, **kwargs: []
    numpy.save = lambda *args, **kwargs: None
    numpy.linalg = types.SimpleNamespace(norm=lambda *args, **kwargs: 1.0)
    sys.modules["numpy"] = numpy

    requests = types.ModuleType("requests")

    class DummyResponse:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"data": [{"embedding": [0.1], "index": 0}], "choices": [{"message": {"content": "ok"}}]}

        def iter_lines(self, decode_unicode=True):
            return iter(())

    class DummySession:
        def mount(self, *args, **kwargs):
            return None

        def get(self, *args, **kwargs):
            return DummyResponse()

        def post(self, *args, **kwargs):
            return DummyResponse()

    requests.Session = DummySession
    requests.Response = DummyResponse
    requests.post = lambda *args, **kwargs: DummyResponse()
    requests.get = lambda *args, **kwargs: DummyResponse()
    requests.patch = lambda *args, **kwargs: DummyResponse()
    requests.ConnectionError = Exception
    requests.ChunkedEncodingError = Exception
    requests.exceptions = types.SimpleNamespace(Timeout=TimeoutError)
    sys.modules["requests"] = requests

    requests_adapters = types.ModuleType("requests.adapters")

    class HTTPAdapter:
        def __init__(self, *args, **kwargs):
            pass

    requests_adapters.HTTPAdapter = HTTPAdapter
    sys.modules["requests.adapters"] = requests_adapters

    urllib3 = types.ModuleType("urllib3")
    urllib3_util = types.ModuleType("urllib3.util")
    urllib3_retry = types.ModuleType("urllib3.util.retry")

    class Retry:
        def __init__(self, *args, **kwargs):
            pass

    urllib3_retry.Retry = Retry
    sys.modules["urllib3"] = urllib3
    sys.modules["urllib3.util"] = urllib3_util
    sys.modules["urllib3.util.retry"] = urllib3_retry

    flask = types.ModuleType("flask")

    class DummyFlask:
        def __init__(self, *args, **kwargs):
            self.url_map = types.SimpleNamespace(iter_rules=lambda: [])

        def route(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def after_request(self, func):
            return func

    flask.Flask = DummyFlask
    flask.Response = object
    flask.jsonify = lambda obj=None, **kwargs: obj if obj is not None else kwargs
    flask.request = types.SimpleNamespace(headers={}, remote_addr="127.0.0.1", method="GET", is_json=False)
    flask.stream_with_context = lambda func: func
    sys.modules["flask"] = flask

    caldav = types.ModuleType("caldav")
    caldav.DAVClient = object
    sys.modules["caldav"] = caldav


class GatewayRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_stub_modules()
        cls.tempdir = tempfile.TemporaryDirectory()
        os.environ["DB_PATH"] = os.path.join(cls.tempdir.name, "test_chats.db")

        cls.database = importlib.import_module("database")
        cls.embedding = importlib.import_module("embedding")
        cls.memory_cards = importlib.import_module("memory_cards")
        cls.memory_tools = importlib.import_module("memory_tools")
        cls.gateway = importlib.import_module("gateway")

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_require_auth_blocks_remote_when_token_missing(self):
        self.gateway.GATEWAY_AUTH_TOKEN = ""
        self.gateway.request = types.SimpleNamespace(headers={}, remote_addr="8.8.8.8")
        protected = self.gateway.require_auth(lambda: "ok")
        result = protected()
        self.assertEqual(result[1], 503)
        self.assertEqual(result[0]["error"], "Gateway auth token is not configured")

    def test_build_like_conditions_use_and_logic(self):
        sql, params = self.database._build_like_conditions(["机器", "学习"])
        self.assertEqual(sql, "content LIKE ? AND content LIKE ?")
        self.assertEqual(params, ["%机器%", "%学习%"])

    def test_build_card_payload_extracts_tags_and_ranges(self):
        fake_messages = [
            {"id": 11, "conversation_id": "c1", "role": "user", "content": "今天胃不舒服"},
            {"id": 12, "conversation_id": "c1", "role": "assistant", "content": "记得吃药"},
        ]
        with mock.patch.object(self.memory_cards, "get_messages_by_date", return_value=fake_messages), \
                mock.patch.object(self.memory_cards, "_call_summary_api", return_value="[每日记忆档案]\n标签: 吃药,胃疼"):
            payload = self.memory_cards._build_card_payload("2026-03-16")
        self.assertEqual(payload["date"], "2026-03-16")
        self.assertEqual(payload["tags"], "吃药,胃疼")
        self.assertEqual(payload["msg_id_start"], 11)
        self.assertEqual(payload["msg_id_end"], 12)

    def test_execute_search_memory_formats_card_tags(self):
        with mock.patch("embedding.get_embedding_for_query", return_value=[0.1]), \
                mock.patch("memory_cards.search_memory_cards", return_value=[{
                    "id": 1,
                    "date": "2026-03-16",
                    "score": 0.95,
                    "summary": "她今天胃疼，记得吃药。",
                    "tags": "吃药,胃疼",
                }]), \
                mock.patch.object(self.gateway, "vector_search_memories", return_value=[]), \
                mock.patch.object(self.memory_tools, "logger"):
            with mock.patch("database.search_history", return_value=[]):
                output = self.memory_tools.execute_search_memory("她今天怎么了")
        self.assertIn("#吃药,胃疼", output)
        self.assertIn("她今天胃疼", output)

    def test_build_messages_keeps_recent_raw_window_and_contextual_query(self):
        incoming = []
        for idx in range(15):
            incoming.append({"role": "user", "content": f"user-{idx}"})
            incoming.append({"role": "assistant", "content": f"assistant-{idx}"})

        with mock.patch.object(self.gateway, "load_system_prompt", return_value="sys {{RECENT_SUMMARY}}"), \
                mock.patch.object(self.gateway, "build_recent_context", return_value="rolling-summary"), \
                mock.patch.object(self.gateway, "_model_specific_patch", return_value=""), \
                mock.patch.object(self.gateway, "_model_max_context", return_value=100000), \
                mock.patch.object(self.gateway, "_model_reliable_tool_calling", return_value=False), \
                mock.patch.object(self.gateway, "extract_search_query", return_value="contextual query") as query_mock, \
                mock.patch.object(self.gateway, "execute_search_memory", return_value="历史记忆片段") as search_mock, \
                mock.patch.object(self.gateway, "_has_recent_overlap", return_value=False):
            final_messages = self.gateway.build_messages(incoming, "glm-4", conversation_id="conv-a")

        non_system = [m for m in final_messages if m["role"] != "system"]
        self.assertLessEqual(len(non_system), self.gateway.RECENT_RAW_MAX_MESSAGES)
        self.assertEqual(non_system[0]["content"], "user-5")
        query_mock.assert_called_once()
        search_mock.assert_called_once_with("contextual query")

    def test_strip_inner_monologue_returns_reply_only(self):
        text = "<inner_monologue>不要泄露</inner_monologue><reply>正常回复</reply>"
        self.assertEqual(self.gateway._strip_inner_monologue(text), "正常回复")

    def test_derive_weekly_digests_from_cards_uses_daily_card_sections(self):
        cards = [
            {
                "date": "2026-03-10",
                "summary": "[每日记忆档案]\n- 【淘淘情绪脉络】: 她有点委屈\n- 【新增约定承诺】: 周末去看海\n- 【状态偏好变化】: 更想被陪着",
            },
            {
                "date": "2026-03-12",
                "summary": "[每日记忆档案]\n- 【淘淘情绪脉络】: 后来放松下来\n- 【关键数值细节】: UID12345\n- 【专属甜蜜回忆】: 一起聊天到很晚",
            },
        ]
        digests = self.memory_cards.derive_weekly_digests_from_cards(cards, max_weeks=1)
        self.assertEqual(len(digests), 1)
        summary = digests[0]["summary"]
        self.assertIn("[每周摘要", summary)
        self.assertIn("周末去看海", summary)
        self.assertIn("UID12345", summary)

    def test_cards_status_reports_unified_summary_architecture(self):
        with mock.patch.object(self.gateway, "_get_card_rebuild_state", return_value={"running": False}), \
                mock.patch.object(self.gateway, "card_vector_store", types.SimpleNamespace(size=3)), \
                mock.patch("database.get_card_count", return_value={"total": 2, "embedded": 2, "dates": 2}):
            status = self.gateway.cards_status()
        self.assertEqual(status["summary_architecture"]["daily_source"], "memory_cards")
        self.assertEqual(status["summary_architecture"]["weekly_source"], "derived_from_memory_cards")
        self.assertFalse(status["summary_architecture"]["legacy_tables_active"])


if __name__ == "__main__":
    unittest.main()
