import os


class AuthConfig:
    AUTOTIM_USERNAME = os.getenv("AUTOTIM_USERNAME", 'admin')
    AUTOTIM_PASSWORD = os.getenv("AUTOTIM_PASSWORD", 'admin')
