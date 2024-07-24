"""Module with specific helper functions to manage the logging of orthoseg.

TODO: maybe it is cleaner to replace most code here by a config dict?
"""

from email.message import EmailMessage
import logging
import smtplib
from typing import Optional

from orthoseg.helpers import config_helper as conf

# Get a logger...
logger = logging.getLogger(__name__)


def sendmail(subject: str, body: Optional[str] = None, stop_on_error: bool = False):
    """Send an email.

    Args:
        subject (str): subject of the email
        body (Optional[str], optional): body of the email. Defaults to None.
        stop_on_error (bool, optional): True to stop when an error occurs sending the
            email. Defaults to False.
    """
    if conf is None:
        raise ValueError("Config is not initialized")

    if not conf.email.getboolean("enabled", fallback=False):
        return
    mail_from = conf.email.get("from", None)
    mail_to = conf.email.get("to", None)
    mail_server = conf.email.get("server", None)
    mail_server_username = conf.email.get("username", None)
    mail_server_password = conf.email.get("password", None)

    # If one of the necessary parameters not provided, log subject
    if mail_from is None or mail_to is None or mail_server is None:
        logger.warning(
            f"Mail global_config not provided to send email with subject: {subject}"
        )
        return

    try:
        # Create message
        msg = EmailMessage()
        msg.add_header("from", mail_from)
        msg.add_header("to", mail_to)
        msg.add_header("subject", subject)
        if body is not None:
            msg.set_payload(body)
            msg.add_header("Content-Type", "text/html")

        # Send the email
        server = smtplib.SMTP(mail_server)
        if mail_server_username is not None and mail_server_password is not None:
            server.login(mail_server_username, mail_server_password)
        server.send_message(msg)
        server.quit()
    except Exception as ex:
        if stop_on_error is False:
            logger.exception("Error sending email")
        else:
            raise RuntimeError("Error sending email") from ex
