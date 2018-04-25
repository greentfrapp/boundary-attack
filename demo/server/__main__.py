#!/usr/bin/env python3

import connexion

from server import encoder
from flask_cors import CORS, cross_origin


def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Boundary Attack API'})
    cors = CORS(app.app)
    app.run(port=8080)


if __name__ == '__main__':
    main()
