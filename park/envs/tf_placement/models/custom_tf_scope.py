import tensorflow as tf

class CustomTFScope(tf.variable_scope):

  o_scopes = []
  all_scopes = []

  def __init__(self, tf_scope, default_name='', *args, **kwargs):
    super(CustomTFScope, self).__init__(tf_scope, default_name, *args, **kwargs)
    self.sc_name = default_name if tf_scope is None else tf_scope

  def __enter__(self):
    #ttysetattr etc goes here before opening and returning the file object
    super(CustomTFScope, self).__enter__()
    CustomTFScope.o_scopes.append(self.sc_name)

  def __exit__(self, type, value, traceback):
    #Exception handling here
    super(CustomTFScope, self).__exit__(type, value, traceback)
    # pop last element
    CustomTFScope.all_scopes.append(CustomTFScope.get_scope_name())
    del CustomTFScope.o_scopes[-1]

  @staticmethod
  def get_scope_name():
    return '/'.join(CustomTFScope.o_scopes)