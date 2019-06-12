#!/usr/bin/python

import storage.api as stapi
import redis

def main():
    r = redis.StrictRedis(host='localhost', port=6379, db=0)
    keys = r.keys()
    print("Number of keys: " + str(len(keys)))
    stapi.redis_connection = r
    for k in keys:
        v = stapi.getByID(k)
        print("KEY: " + str(k) + "  VALUE: " + str(v))

if __name__ == '__main__':
    main()
