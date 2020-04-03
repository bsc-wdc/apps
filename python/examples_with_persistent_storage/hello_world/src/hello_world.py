from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from utils.classes import hello


@task(returns=1)
def create_greeting(message):
    """
    Instantiates a persistent object and populates it with the received
    message.
    :param message: String with the information to store in the psco.
    :return: The populated persistent object.
    """
    hi = hello(message)
    hi.make_persistent("greet")
    return hi


@task(returns=1)
def greet(greetings):
    """
    Retrieves the information contained in the given persistent object.
    :param greetings: Persistent object.
    :return: String with the psco content.
    """
    content = greetings.get()
    return content


@task(returns=1)
def check_greeting(content, message):
    """
    Checcks that the given content is equal to the given message.
    :param content: String with content.
    :param message: String with message.
    :return: Boolean (True if equal, False otherwise).
    """
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
        msg = "SOMETHING FAILED!!!"
        print(msg)
        raise Exception(msg)


if __name__ == '__main__':
    main()
