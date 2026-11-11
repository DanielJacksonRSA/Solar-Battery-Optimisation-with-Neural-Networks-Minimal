# Copyright (c) 2026 Daniel Jackson and Nelson Mandela University. All rights reserved.
# This code is proprietary. It may not be used, copied, modified, distributed or
# reused in whole or in part, nor may the ideas it embodies be used, without the
# prior written permission of the copyright holders. See LICENSE.
from __future__ import annotations
import argparse
import uvicorn

def parse_args() -> argparse.Namespace:
    var1 = argparse.ArgumentParser(description='Run the solar/battery system simulation web server.')
    var1.add_argument('--host', default='127.0.0.1')
    var1.add_argument('--port', type=int, default=8000)
    var1.add_argument('--reload', action='store_true')
    return var1.parse_args()

def main() -> None:
    var2 = parse_args()
    uvicorn.run('api:app', host=var2.host, port=var2.port, reload=var2.reload)
if __name__ == '__main__':
    main()
