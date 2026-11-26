"""
Operational emails to the developer (Try-On Studio B13, 2026-11-26). Never sent to users.

Plain SMTP, configured only in .env (nothing secret in code or logs):
    ALERT_EMAIL_TO   where alerts go (your address)
    SMTP_HOST        e.g. smtp.gmail.com
    SMTP_PORT        465 (SSL, default) or 587 (STARTTLS)
    SMTP_USER        the sending account
    SMTP_PASSWORD    its app password (Gmail: Google Account -> Security -> App passwords)
    SMTP_FROM        optional, defaults to SMTP_USER

Not configured -> the alert is only logged (and still shows on the Developer screen).
"""

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)

REQUIRED = ("ALERT_EMAIL_TO", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")


def email_configured() -> bool:
    return all((os.getenv(k) or "").strip() for k in REQUIRED)


def send_ops_email(subject: str, body: str) -> str:
    """Returns "sent", "not_configured" or "failed: <ErrorType>". Never raises."""
    if not email_configured():
        logger.warning("ops email not configured - alert only logged: %s", subject)
        return "not_configured"
    host = os.getenv("SMTP_HOST").strip()
    user = os.getenv("SMTP_USER").strip()
    password = os.getenv("SMTP_PASSWORD").strip()
    try:
        port = int(os.getenv("SMTP_PORT") or 465)
    except ValueError:
        port = 465
    msg = EmailMessage()
    msg["Subject"] = f"[FitCheck ops] {subject}"
    msg["From"] = (os.getenv("SMTP_FROM") or user).strip()
    msg["To"] = os.getenv("ALERT_EMAIL_TO").strip()
    msg.set_content(body)
    try:
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls(context=context)
                smtp.login(user, password)
                smtp.send_message(msg)
        logger.info("ops email sent: %s", subject)
        return "sent"
    except Exception as exc:                       # never log the password or server reply
        logger.warning("ops email failed (%s): %s", type(exc).__name__, subject)
        return f"failed: {type(exc).__name__}"