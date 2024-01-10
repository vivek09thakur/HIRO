import secrets


def generate_random_string(length=32):
    return secrets.token_urlsafe(length)
