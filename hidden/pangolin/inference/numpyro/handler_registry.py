from pangolin import util

#numpyro_handlers = util.WriteOnceDict()
# TODO: some trouble with repeated imports causing things to get registered more than once
numpyro_handlers = {}

def register_handler(op_class):    
    def doit(handle_fun):
        numpyro_handlers[op_class] = handle_fun
    return doit
