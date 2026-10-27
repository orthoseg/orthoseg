from email import policy
from email.parser import BytesParser
from types import SimpleNamespace

from orthoseg.helpers import email_helper


def test_sendmail_encodes_unicode_html_body(monkeypatch):
    settings = {
        "from": "sender@example.com",
        "to": "recipient@example.com",
        "server": "mail.example.com",
        "username": None,
        "password": None,
    }

    class EmailConfig:
        def getboolean(self, key, fallback=False):
            return key == "enabled" or fallback

        def get(self, key, fallback=None):
            return settings.get(key, fallback)

    sent_messages = []

    class SmtpServer:
        def __init__(self, server):
            pass

        def send_message(self, message):
            sent_messages.append(message)

        def quit(self):
            pass

    monkeypatch.setattr(
        email_helper,
        "conf",
        SimpleNamespace(email=EmailConfig()),
    )
    monkeypatch.setattr(email_helper.smtplib, "SMTP", SmtpServer)

    email_helper.sendmail(
        subject="Prediction failed",
        body="GPU error • retry failed",
        stop_on_error=True,
    )

    message = BytesParser(policy=policy.default).parsebytes(sent_messages[0].as_bytes())
    assert message.get_content_type() == "text/html"
    assert message.get_content_charset() == "utf-8"
    assert message.get_content() == "GPU error • retry failed\n"
