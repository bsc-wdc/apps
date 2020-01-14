import sys
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from utils.classes import hello

@task(returns=1)
def create_greeting(message):
    hi = hello(message)
    hi.make_persistent()
    return hi

@task(returns=1)
def greet(greetings):
    content = greetings.get()
    return content

@task(returns=1)
def check_greeting(content, message):
    return content == message

def main():
    message = "Hello world"
    greeting = create_greeting(message)
    content = greet(greeting)
    result = check_greeting(content, message)
    result_wrong = check_greeting(content, message + "!!!")
    result = compss_wait_on(result)
    result_wrong = compss_wait_on(result_wrong)
    if result != result_wrong:
        print("THE RESULT IS OK")
    else:
        print("SOMETHING FAILED!!!")


if __name__ == '__main__':
    main()
