import logging
import functools


# Configuring the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_to_termianal(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function {func.__name__} with arguments: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Function {func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} raised an exception: {e}")
            raise
    return wrapper




# test function 
@log_to_termianal
def add(a,b):
    logging.info("Getting two values to add")
    c = a + b 
    logging.info('Added two values and returning the sum ')
    return c 




if __name__ == '__main__':
    log_to_termianal(add(3,2))