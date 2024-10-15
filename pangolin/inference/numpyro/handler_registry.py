from pangolin import util

numpyro_handlers = util.WriteOnceDict()

def register_handler(op_class):
    def doit(handle_fun):
        numpyro_handlers[op_class] = handle_fun
    return doit
