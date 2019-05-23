
import smtplib
from email.message import EmailMessage

email_server = "ALV-MAIL-P1.DG3.be"

# Create message
msg = EmailMessage()
msg.add_header('from', "pieter.roggemans@lv.vlaanderen.be")
msg.add_header('to', "pieter.roggemans@lv.vlaanderen.be")
msg.add_header('subject', "Hello!")
msg.set_payload("This is a test of emailing through smtp.")

# Send the mail
server = smtplib.SMTP(email_server)
#server.login("MrDoe", "PASSWORD")
server.send_message(msg)
server.quit()