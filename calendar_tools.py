"""
Calendar tool for AI model function calling.

Writes events to the user's Apple Calendar via CalDAV (iCloud).
Requires environment variables:
  CALDAV_URL      - iCloud CalDAV URL (default: https://caldav.icloud.com)
  CALDAV_USERNAME - Apple ID email
  CALDAV_PASSWORD - App-specific password (NOT the main Apple ID password)
  CALDAV_CALENDAR - Calendar name to write to (default: "小克提醒")
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import caldav

logger = logging.getLogger(__name__)

# ---------- Config ----------
CALDAV_URL = os.getenv("CALDAV_URL", "https://caldav.icloud.com")
CALDAV_USERNAME = os.getenv("CALDAV_USERNAME", "")
CALDAV_PASSWORD = os.getenv("CALDAV_PASSWORD", "")
CALDAV_CALENDAR_NAME = os.getenv("CALDAV_CALENDAR", "小克提醒")

# China Standard Time
CST = timezone(timedelta(hours=8))

# ---------- Tool Definition (OpenAI function calling format) ----------

CALENDAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": (
                "将日程事件写入淘淘的手机日历。"
                "当她提到未来的安排、吃药时间、会议或纪念日时使用。"
                "也可以在判断她需要被提醒时主动调用（主动关怀）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "事件标题，例如「吃药」「和闺蜜吃饭」「纪念日」",
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "开始时间，ISO 8601 格式，例如 2025-03-15T09:00:00+08:00。"
                            "如果淘淘只说了日期没说时间，默认用 09:00。"
                        ),
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "结束时间，ISO 8601 格式。可选。"
                            "如果不提供，默认为 start_time + 1小时。"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "备注信息，可选。",
                    },
                },
                "required": ["title", "start_time"],
            },
        },
    },
]


# ---------- CalDAV Client ----------

def _get_calendar():
    """Connect to CalDAV server and return the target calendar object."""
    if not CALDAV_USERNAME or not CALDAV_PASSWORD:
        raise RuntimeError("CalDAV credentials not configured (CALDAV_USERNAME / CALDAV_PASSWORD)")

    client = caldav.DAVClient(
        url=CALDAV_URL,
        username=CALDAV_USERNAME,
        password=CALDAV_PASSWORD,
    )
    principal = client.principal()
    calendars = principal.calendars()

    # Find the target calendar by name
    for cal in calendars:
        if cal.name == CALDAV_CALENDAR_NAME:
            logger.info(f"[Calendar] found calendar: {cal.name}")
            return cal

    # If target calendar doesn't exist, try to create it
    logger.info(f"[Calendar] calendar '{CALDAV_CALENDAR_NAME}' not found, creating...")
    try:
        cal = principal.make_calendar(name=CALDAV_CALENDAR_NAME)
        logger.info(f"[Calendar] created calendar: {CALDAV_CALENDAR_NAME}")
        return cal
    except Exception as e:
        logger.warning(f"[Calendar] failed to create calendar: {e}")
        # Fall back to the first available calendar
        if calendars:
            logger.info(f"[Calendar] falling back to: {calendars[0].name}")
            return calendars[0]
        raise RuntimeError("No calendars available on this CalDAV account")


def _parse_time(time_str: str) -> datetime:
    """Parse an ISO 8601 time string, tolerating common variations."""
    # Strip trailing Z and treat as UTC
    if time_str.endswith("Z"):
        time_str = time_str[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(time_str)
    except ValueError:
        pass

    # Try common formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(tzinfo=CST)  # assume CST if no tz
        except ValueError:
            continue

    raise ValueError(f"Cannot parse time: {time_str}")


# ---------- Tool Execution ----------

def execute_add_calendar_event(arguments: dict) -> str:
    """
    Execute add_calendar_event tool call.
    Creates an event on the user's calendar via CalDAV.
    Returns a status message for the model.
    """
    title = arguments.get("title", "").strip()
    start_str = arguments.get("start_time", "").strip()
    end_str = arguments.get("end_time", "").strip()
    description = arguments.get("description", "").strip()

    if not title:
        return json.dumps({"success": False, "error": "缺少事件标题"}, ensure_ascii=False)
    if not start_str:
        return json.dumps({"success": False, "error": "缺少开始时间"}, ensure_ascii=False)

    try:
        start_dt = _parse_time(start_str)
    except ValueError as e:
        return json.dumps({"success": False, "error": f"开始时间格式错误: {e}"}, ensure_ascii=False)

    if end_str:
        try:
            end_dt = _parse_time(end_str)
        except ValueError:
            end_dt = start_dt + timedelta(hours=1)
    else:
        end_dt = start_dt + timedelta(hours=1)

    # Ensure timezone
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=CST)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=CST)

    # Build VCALENDAR
    # Use VALARM for a 15-minute reminder
    vevent_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//XiaoKe//Calendar//CN",
        "BEGIN:VEVENT",
        f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%S')}",
        f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%S')}",
        f"SUMMARY:{title}",
    ]
    if description:
        # Escape newlines for iCal format
        safe_desc = description.replace("\n", "\\n").replace(",", "\\,")
        vevent_lines.append(f"DESCRIPTION:{safe_desc}")

    # Add a 15-minute reminder alarm
    vevent_lines.extend([
        "BEGIN:VALARM",
        "TRIGGER:-PT15M",
        "ACTION:DISPLAY",
        f"DESCRIPTION:{title}",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
    ])
    vcal_str = "\r\n".join(vevent_lines)

    try:
        cal = _get_calendar()
        cal.save_event(vcal_str)
        logger.info(f"[Calendar] event created: '{title}' at {start_dt.isoformat()}")
        return json.dumps({
            "success": True,
            "title": title,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "message": "日程已写入日历",
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"[Calendar] failed to create event: {e}", exc_info=True)
        return json.dumps({
            "success": False,
            "error": f"写入日历失败: {str(e)}",
        }, ensure_ascii=False)
