import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject, body, to_email):
    from_email = os.getenv("ps12344@gmail.com")  # Fetch email address from environment variable
    from_password = os.getenv("Sk@1")  # Fetch email password from environment variable

    # Create the email
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = subject

    # Attach the email body
    message.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(from_email, from_password)  # Login to the server

        # Send the email
        server.send_message(message)
        server.quit()  # Close the connection

        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")


# Example usage
subject = "Test Email"
body = "This is a test email sent from Python."
to_email = "sk7668king@gmail.com"  # Replace with recipient email address

send_email(subject, body, to_email)
