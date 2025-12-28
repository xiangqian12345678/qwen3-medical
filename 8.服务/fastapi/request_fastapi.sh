#!/bin/bash

curl -X 'POST' \
  'http://localhost:8801/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "特朗普是谁"
}'