from typing import cast

from fastapi import Request

from urbanflow.api.services import ApiServices


def get_services(request: Request) -> ApiServices:
    return cast(ApiServices, request.app.state.services)
