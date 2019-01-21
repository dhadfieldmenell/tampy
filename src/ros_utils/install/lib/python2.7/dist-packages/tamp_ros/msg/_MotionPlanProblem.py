# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from tamp_ros/MotionPlanProblem.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import std_msgs.msg

class MotionPlanProblem(genpy.Message):
  _md5sum = "62f92843a78550529c22bfbaad9b886c"
  _type = "tamp_ros/MotionPlanProblem"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """int32 solver_id
int32 prob_id
int32 server_id
string task
float32[] state
int32 cond
std_msgs/Float32MultiArray[] traj_mean

bool use_prior
float32[] sigma
float32[] mu
float32[] logmass
float32[] mass
int32 N
int32 K
int32 Do

================================================================================
MSG: std_msgs/Float32MultiArray
# Please look at the MultiArrayLayout message definition for
# documentation on all multiarrays.

MultiArrayLayout  layout        # specification of data layout
float32[]         data          # array of data


================================================================================
MSG: std_msgs/MultiArrayLayout
# The multiarray declares a generic multi-dimensional array of a
# particular data type.  Dimensions are ordered from outer most
# to inner most.

MultiArrayDimension[] dim # Array of dimension properties
uint32 data_offset        # padding elements at front of data

# Accessors should ALWAYS be written in terms of dimension stride
# and specified outer-most dimension first.
# 
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
#
# A standard, 3-channel 640x480 image with interleaved color channels
# would be specified as:
#
# dim[0].label  = "height"
# dim[0].size   = 480
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
# dim[1].label  = "width"
# dim[1].size   = 640
# dim[1].stride = 3*640 = 1920
# dim[2].label  = "channel"
# dim[2].size   = 3
# dim[2].stride = 3
#
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.

================================================================================
MSG: std_msgs/MultiArrayDimension
string label   # label of given dimension
uint32 size    # size of given dimension (in type units)
uint32 stride  # stride of given dimension"""
  __slots__ = ['solver_id','prob_id','server_id','task','state','cond','traj_mean','use_prior','sigma','mu','logmass','mass','N','K','Do']
  _slot_types = ['int32','int32','int32','string','float32[]','int32','std_msgs/Float32MultiArray[]','bool','float32[]','float32[]','float32[]','float32[]','int32','int32','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       solver_id,prob_id,server_id,task,state,cond,traj_mean,use_prior,sigma,mu,logmass,mass,N,K,Do

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(MotionPlanProblem, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.solver_id is None:
        self.solver_id = 0
      if self.prob_id is None:
        self.prob_id = 0
      if self.server_id is None:
        self.server_id = 0
      if self.task is None:
        self.task = ''
      if self.state is None:
        self.state = []
      if self.cond is None:
        self.cond = 0
      if self.traj_mean is None:
        self.traj_mean = []
      if self.use_prior is None:
        self.use_prior = False
      if self.sigma is None:
        self.sigma = []
      if self.mu is None:
        self.mu = []
      if self.logmass is None:
        self.logmass = []
      if self.mass is None:
        self.mass = []
      if self.N is None:
        self.N = 0
      if self.K is None:
        self.K = 0
      if self.Do is None:
        self.Do = 0
    else:
      self.solver_id = 0
      self.prob_id = 0
      self.server_id = 0
      self.task = ''
      self.state = []
      self.cond = 0
      self.traj_mean = []
      self.use_prior = False
      self.sigma = []
      self.mu = []
      self.logmass = []
      self.mass = []
      self.N = 0
      self.K = 0
      self.Do = 0

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
      _x = self
      buff.write(_get_struct_3i().pack(_x.solver_id, _x.prob_id, _x.server_id))
      _x = self.task
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.state)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.state))
      buff.write(_get_struct_i().pack(self.cond))
      length = len(self.traj_mean)
      buff.write(_struct_I.pack(length))
      for val1 in self.traj_mean:
        _v1 = val1.layout
        length = len(_v1.dim)
        buff.write(_struct_I.pack(length))
        for val3 in _v1.dim:
          _x = val3.label
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          buff.write(struct.pack('<I%ss'%length, length, _x))
          _x = val3
          buff.write(_get_struct_2I().pack(_x.size, _x.stride))
        buff.write(_get_struct_I().pack(_v1.data_offset))
        length = len(val1.data)
        buff.write(_struct_I.pack(length))
        pattern = '<%sf'%length
        buff.write(struct.pack(pattern, *val1.data))
      buff.write(_get_struct_B().pack(self.use_prior))
      length = len(self.sigma)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.sigma))
      length = len(self.mu)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.mu))
      length = len(self.logmass)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.logmass))
      length = len(self.mass)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(struct.pack(pattern, *self.mass))
      _x = self
      buff.write(_get_struct_3i().pack(_x.N, _x.K, _x.Do))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.traj_mean is None:
        self.traj_mean = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.solver_id, _x.prob_id, _x.server_id,) = _get_struct_3i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.task = str[start:end].decode('utf-8')
      else:
        self.task = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.state = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (self.cond,) = _get_struct_i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.traj_mean = []
      for i in range(0, length):
        val1 = std_msgs.msg.Float32MultiArray()
        _v2 = val1.layout
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        _v2.dim = []
        for i in range(0, length):
          val3 = std_msgs.msg.MultiArrayDimension()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val3.label = str[start:end].decode('utf-8')
          else:
            val3.label = str[start:end]
          _x = val3
          start = end
          end += 8
          (_x.size, _x.stride,) = _get_struct_2I().unpack(str[start:end])
          _v2.dim.append(val3)
        start = end
        end += 4
        (_v2.data_offset,) = _get_struct_I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        pattern = '<%sf'%length
        start = end
        end += struct.calcsize(pattern)
        val1.data = struct.unpack(pattern, str[start:end])
        self.traj_mean.append(val1)
      start = end
      end += 1
      (self.use_prior,) = _get_struct_B().unpack(str[start:end])
      self.use_prior = bool(self.use_prior)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.sigma = struct.unpack(pattern, str[start:end])
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
      self.logmass = struct.unpack(pattern, str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.mass = struct.unpack(pattern, str[start:end])
      _x = self
      start = end
      end += 12
      (_x.N, _x.K, _x.Do,) = _get_struct_3i().unpack(str[start:end])
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
      _x = self
      buff.write(_get_struct_3i().pack(_x.solver_id, _x.prob_id, _x.server_id))
      _x = self.task
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      length = len(self.state)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.state.tostring())
      buff.write(_get_struct_i().pack(self.cond))
      length = len(self.traj_mean)
      buff.write(_struct_I.pack(length))
      for val1 in self.traj_mean:
        _v3 = val1.layout
        length = len(_v3.dim)
        buff.write(_struct_I.pack(length))
        for val3 in _v3.dim:
          _x = val3.label
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          buff.write(struct.pack('<I%ss'%length, length, _x))
          _x = val3
          buff.write(_get_struct_2I().pack(_x.size, _x.stride))
        buff.write(_get_struct_I().pack(_v3.data_offset))
        length = len(val1.data)
        buff.write(_struct_I.pack(length))
        pattern = '<%sf'%length
        buff.write(val1.data.tostring())
      buff.write(_get_struct_B().pack(self.use_prior))
      length = len(self.sigma)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.sigma.tostring())
      length = len(self.mu)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.mu.tostring())
      length = len(self.logmass)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.logmass.tostring())
      length = len(self.mass)
      buff.write(_struct_I.pack(length))
      pattern = '<%sf'%length
      buff.write(self.mass.tostring())
      _x = self
      buff.write(_get_struct_3i().pack(_x.N, _x.K, _x.Do))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.traj_mean is None:
        self.traj_mean = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.solver_id, _x.prob_id, _x.server_id,) = _get_struct_3i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.task = str[start:end].decode('utf-8')
      else:
        self.task = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.state = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (self.cond,) = _get_struct_i().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.traj_mean = []
      for i in range(0, length):
        val1 = std_msgs.msg.Float32MultiArray()
        _v4 = val1.layout
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        _v4.dim = []
        for i in range(0, length):
          val3 = std_msgs.msg.MultiArrayDimension()
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            val3.label = str[start:end].decode('utf-8')
          else:
            val3.label = str[start:end]
          _x = val3
          start = end
          end += 8
          (_x.size, _x.stride,) = _get_struct_2I().unpack(str[start:end])
          _v4.dim.append(val3)
        start = end
        end += 4
        (_v4.data_offset,) = _get_struct_I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        pattern = '<%sf'%length
        start = end
        end += struct.calcsize(pattern)
        val1.data = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
        self.traj_mean.append(val1)
      start = end
      end += 1
      (self.use_prior,) = _get_struct_B().unpack(str[start:end])
      self.use_prior = bool(self.use_prior)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.sigma = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
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
      self.logmass = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      pattern = '<%sf'%length
      start = end
      end += struct.calcsize(pattern)
      self.mass = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
      _x = self
      start = end
      end += 12
      (_x.N, _x.K, _x.Do,) = _get_struct_3i().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_i = None
def _get_struct_i():
    global _struct_i
    if _struct_i is None:
        _struct_i = struct.Struct("<i")
    return _struct_i
_struct_3i = None
def _get_struct_3i():
    global _struct_3i
    if _struct_3i is None:
        _struct_3i = struct.Struct("<3i")
    return _struct_3i
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
