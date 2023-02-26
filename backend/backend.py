import os
import logging
from flask import Flask, request, jsonify
from dds_utils import ServerConfig
import json
import yaml
from .server import Server
import time
from datetime import datetime

app = Flask(__name__)
server = None

from munch import *


@app.route("/")
@app.route("/index")
def index():
    # TODO: Add debugging information to the page if needed
    return "Much to do!"


@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.load(request.data, Loader=yaml.SafeLoader)
    global server
    if not server:
        logging.basicConfig(
            format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
            level="INFO")
        server = Server(args, args["nframes"])
        os.makedirs("server_temp", exist_ok=True)
        os.makedirs("server_temp-cropped", exist_ok=True)
        return "New Init"
    else:
        server.reset_state(int(args["nframes"]))
        return "Reset"

@app.route("/low", methods=["POST"])
def low_query():
    t1 = datetime.now()
    print('1 begin read data---------------------------------------', t1)
    # fo=open('server.txt','a+')
    # fo.write(f'1 receive data:{t1}\n')
    file_data = request.files["media"]
    t1=datetime.now()
    print('1 end receive data------------------------------------------', t1)
    print('-------------------------------------------------------------')
    print('1 begin perform_low_query-----------------------------------',t1)
    results = server.perform_low_query(file_data)   # results include rpns
    t2 = datetime.now()
    print('1 end perform_low_query-----------------------------------', t2,'--',t2-t1)
    # fo.write(f'1 perform_low_query total time:{t2-t1}\n')
    print('------------------------------------------------------------')
    print('1 begin send results-----------------------------------', t2)
    # fo.write(f'1 begin send results:{t2}\n')
    # fo.close()
    return jsonify(results)


@app.route("/high", methods=["POST"])
def high_query():
    t1 = datetime.now()
    print('2 begin read data---------------------------------------', t1)
    # fo = open('server.txt', 'a+')
    # fo.write(f'2 receive data:{t1}\n')
    file_data = request.files["media"]
    t1 = datetime.now()
    print('2 end receive data------------------------------------------', t1)
    print('-------------------------------------------------------------')
    print('2 begin perform_high_query-----------------------------------', t1)
    results = server.perform_high_query(file_data)
    t2 = datetime.now()
    print('2 end perform_high_query-----------------------------------', t2, '--', t2 - t1)
    # fo.write(f'2 perform_high_query total time:{t2-t1}\n')
    print('------------------------------------------------------------')
    print('2 begin send results-----------------------------------', t2)
    # fo.write(f'2 begin send results:{t2}\n')
    # fo.close()
    return jsonify(results)
