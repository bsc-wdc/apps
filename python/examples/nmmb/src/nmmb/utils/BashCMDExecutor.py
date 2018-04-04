import os
import logging
import subprocess

from userexceptions.CommandException import CommandException
from loggers import LoggerNames


class BashCMDExecutor(object):
    """
    Helper class to execute bash commands
    """

    LOGGER = logging.getLogger(LoggerNames.BASH_CMD)

    ERROR_PROC_EXEC = "[ERROR] Exception executing command"

    command = None
    arguments = []
    redirectoutput = None

    '''
    * Creates a Bash Executor for the given command @command
    *
    * @param command
    '''
    def __init__(self, command):
        self.command = command
        self.arguments = []
        self.redirectoutput = None

    '''
    * Adds an argument @arg to the command
    *
    * @param arg
    '''
    def addArgument(self, arg):
        self.arguments.append(arg)

    '''
    * Adds a flag and its value in two separated arguments to the command
    *
    * @param flag
    * @param value
    '''
    def addFlagAndValue(self, flag, value):
        self.arguments.append(flag)
        self.arguments.append(value)

    '''
    * Redirects the output to the specified filePath
    *
    * @param filePath
    '''
    def redirectOutput(self, filePath):
        self.redirectoutput = filePath

    '''
    * Executes the command with all the specified parameters and output redirections
    *
    * @return
    * @throws CommandException
    '''
    def execute(self):
        self.LOGGER.info("[CMD EXECUTION WRAPPER] Executing command: " + self.__str__())

        # Prepare command execution
        cmd = [self.command]
        cmd += self.arguments

        # if "LD_PRELOAD" in os.environ:
        #     del os.environ["LD_PRELOAD"]

        stdoutf = subprocess.PIPE
        # Add redirection if needed
        if self.redirectoutput is not None:
            stdoutf = open(self.redirectoutput, 'wb')

        # Launch command
        out = ""
        err = ""
        exitValue = -1
        try:
            process = subprocess.Popen(cmd, stdout=stdoutf, stderr=subprocess.PIPE)
            self.LOGGER.debug("[CMD EXECUTION WRAPPER] Waiting for CMD completion")
            out, err = process.communicate()
            exitValue = process.returncode
            # Log binary execution
            self.logBinaryExecution(out, err, exitValue)
        except Exception as e:
            raise CommandException(self.ERROR_PROC_EXEC + str(e))
        #finally:
        #    # Log binary execution
        #    self.logBinaryExecution(out, err, exitValue)

        self.LOGGER.info("[CMD EXECUTION WRAPPER] End command execution")

        # Return process exit value
        return exitValue

    def logBinaryExecution(self, out, err, exitValue):
        # Print all process execution information
        self.LOGGER.debug("[CMD EXECUTION WRAPPER] ------------------------------------")
        self.LOGGER.debug("[CMD EXECUTION WRAPPER] CMD EXIT VALUE: " + str(exitValue))
        self.LOGGER.debug("[CMD EXECUTION WRAPPER] ------------------------------------")
        if exitValue == 0:
            self.LOGGER.debug("[CMD EXECUTION WRAPPER] CMD OUTPUT:")
            self.LOGGER.debug(out)
            self.LOGGER.error("[CMD EXECUTION WRAPPER] ------------------------------------")
        else:
            self.LOGGER.error("[CMD EXECUTION WRAPPER] CMD ERROR:")
            self.LOGGER.debug(err)
            self.LOGGER.error("[CMD EXECUTION WRAPPER] ------------------------------------")

    def __str__(self):
        sb = self.command
        for arg in self.arguments:
            sb += " "
            sb += arg

        if self.redirectoutput is not None:
            sb += " > "
            sb += self.redirectoutput

        return sb
