import os
import asyncio
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------
# Sends a simple "menu ready" notification email using Google Workspace /
# Gmail SMTP. It only sends a notification — no HTML template, plain
# text, with an optional CSV attachment.
#
# ENV VARS that need to be set in .env / environment:
#
#   SMTP_HOST       -> "smtp.gmail.com" (same for both Google Workspace
#                       and Gmail)
#   SMTP_PORT       -> 587 (TLS)
#   SMTP_USER       -> The Workspace/Gmail email address used to send
#                       emails, e.g. "notifications@yourdomain.com"
#   SMTP_APP_PASSWORD -> Google App Password (16-character). Your normal
#                       account password will NOT work.
#   SMTP_FROM_NAME  -> "Menu Studio" (or any sender name you want to show)
#
# How to create an App Password (if 2-Step Verification is ON, which it
# should be):
#   1) Google Account -> Security -> Turn ON 2-Step Verification
#   2) Google Account -> Security -> "App passwords" (search for
#      "app passwords" if it is not directly visible)
#   3) Enter an app name (e.g. "Menu Studio Backend") -> Generate
#   4) You will receive a 16-character password. Use that value for
#      SMTP_APP_PASSWORD.
#
# If this is a Google Workspace admin-managed account and the
# "App passwords" option is not available, ask the Workspace admin to
# enable the required settings, or use the OAuth2 route instead
# (more complex and not necessary for now).
# -----------------------------------------------------------------------

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Menu Extractor")


def _send_email_sync(
    to_email: str,
    subject: str,
    body: str,
    attachment_bytes: bytes | None = None,
    attachment_filename: str | None = None,
) -> None:
    if not SMTP_USER or not SMTP_APP_PASSWORD:
        print(
            "[email_service] SMTP_USER / SMTP_APP_PASSWORD are not configured — email skipped.")
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_bytes is not None and attachment_filename:
        msg.add_attachment(
            attachment_bytes,
            maintype="text",
            subtype="csv",
            filename=attachment_filename,
        )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_APP_PASSWORD)
        server.send_message(msg)


async def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachment_bytes: bytes | None = None,
    attachment_filename: str | None = None,
) -> None:
    """
    smtplib is blocking, so we run it in a separate thread to avoid
    blocking the event loop. Await this function inside a background task.
    """
    try:
        await asyncio.to_thread(
            _send_email_sync,
            to_email,
            subject,
            body,
            attachment_bytes,
            attachment_filename,
        )
        print(f"[email_service] Email sent to {to_email}")
    except Exception as e:
        # An email failure should not cause the entire pipeline to fail,
        # so we log the error and return quietly.
        print(f"[email_service] Email FAILED to {to_email}: {e}")


# -----------------------------------------------------------------------
# Specific notification helpers that will be called from the router
# -----------------------------------------------------------------------

async def send_menu_ready_email(
    to_email: str,
    restaurant_name: str,
    menu_id: int,
    menu_number: int,
    csv_bytes: bytes | None = None,
) -> None:
    subject = f"Your menu #{menu_number} is ready"
    body = (
        f"Hello {restaurant_name},\n\n"
        f"Your uploaded menu (Menu #{menu_number}, ID: {menu_id}) "
        f"has been successfully processed and is now ready.\n\n"
        f"You can now log in to Menu Studio and review it.\n\n"
        f"Thank you,\nMenu Studio"
    )

    filename = f"menu_{menu_id}_items.csv" if csv_bytes is not None else None

    await send_email(
        to_email,
        subject,
        body,
        attachment_bytes=csv_bytes,
        attachment_filename=filename,
    )


async def send_menu_failed_email(to_email: str, restaurant_name: str, menu_id: int, menu_number: int) -> None:
    subject = f"Menu #{menu_number} could not be processed"
    body = (
        f"Hello {restaurant_name},\n\n"
        f"Unfortunately, we encountered an issue while processing your "
        f"uploaded menu (Menu #{menu_number}, ID: {menu_id}).\n\n"
        f"Please try uploading the menu again, or contact support if the "
        f"problem continues.\n\n"
        f"Thank you,\nMenu Studio"
    )
    await send_email(to_email, subject, body)
