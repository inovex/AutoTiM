from flask import Response


RESPONSE_400_INPUT_FORMAT_WRONG = Response("Wrong input format: this service requires input data "
                                           "to be a single csv-file.", status=400)
RESPONSE_400_NO_REQUIRED_PARAM = Response("One or more of required parameters do not exist.",
                                          status=400)
RESPONSE_500_INTERNAL_SERVER_ERROR = Response(status=500)
