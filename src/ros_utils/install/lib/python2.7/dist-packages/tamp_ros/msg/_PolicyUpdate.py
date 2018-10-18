# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from tamp_ros/PolicyUpdate.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class PolicyUpdate(genpy.Message):
  _md5sum = "74d9a9ad258b0d5854987033dafe686d"
  _type = "tamp_ros/PolicyUpdate"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """float32[] obs
float32[] mu
float32[] prc
float32[] wt

int32 dO
int32 dPrimObs
int32 dU
int32 n
int32 rollout_len
"""
  __slots__ = ['obs','mu','prc','wt','dO','dPrimObs','dU','n','rollout_len']
  _slot_types = ['float32[]','float32[]','float32[]','float32[]','int32','int32','int32','int32','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       obs,mu,prc,wt,dO,dPrimObs,dU,n,rollout_len

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(PolicyUpdate, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.obs is None:
        self.obs = []
      if self.mu is None:
        self.mu = []
      if self.prc is None:
        self.prc = []
      if self.wt is None:
        self.wt = []
      if self.dO is None:
        self.dO = 0
      if self.dPrimObs is None:
        self.dPrimObs = 0
      if self.dU is None:
        self.dU = 0
      if self.n is None:
        self.n = 0
      if self.rollout_len is None:
        self.rollout_len = 0
    else:
      self.obs = []
      self.mu = []
      self.prc = []
      self.wt = []
      self.dO = 0
      self.dPrimObs = 0
      self.dU = 0
      self.n = 0
      self.rollout_len = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      length = len(self.obs)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.obs))
      length = len(self.mu)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.mu))
      length = len(self.prc)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.prc))
      length = len(self.wt)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.wt))
      _x = self
      buff.write(_get_struct_5i().pack(_x.dO, _x.dPrimObs, _x.dU, _x.n, _x.rollout_len))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.obs = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.mu = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.prc = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.wt = struct.unpack(pattern, str[start:end])
      _x = self
      start = end
      end += 20
      (_x.dO, _x.dPrimObs, _x.dU, _x.n, _x.rollout_len,) = _get_struct_5i().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      length = len(self.obs)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.obs.tostring())
      length = len(self.mu)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.mu.tostring())
      length = len(self.prc)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.prc.tostring())
      length = len(self.wt)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.wt.tostring())
      _x = self
      buff.write(_get_struct_5i().pack(_x.dO, _x.dPrimObs, _x.dU, _x.n, _x.rollout_len))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.obs = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.mu = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.prc = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.wt = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      _x = self
      start = end
      end += 20
      (_x.dO, _x.dPrimObs, _x.dU, _x.n, _x.rollout_len,) = _get_struct_5i().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_5i = None
def _get_struct_5i():
    global _struct_5i
    if _struct_5i is None:
        _struct_5i = struct.Struct("<5i")
    return _struct_5i