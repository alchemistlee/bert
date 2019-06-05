#!/usr/bin/env bash

#1. 启动后台服务

nohup python3 spear_server.py --host 0.0.0.0 --port 10003 --authkey qweasd &

#2. 启动服务接口

nohup python3 spear_interface.py > web.log 2>&1 &


