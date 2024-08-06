import secrets
import string


def generate_random_str(len=16):
    random_string = "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(len)
    )
    return random_string
