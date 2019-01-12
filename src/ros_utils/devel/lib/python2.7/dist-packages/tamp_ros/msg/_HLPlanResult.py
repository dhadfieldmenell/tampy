# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from tamp_ros/HLPlanResult.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import tamp_ros.msg
import std_msgs.msg

class HLPlanResult(genpy.Message):
  _md5sum = "3d15420a1ea98d1e6019df3adbc0e259"
  _type = "tamp_ros/HLPlanResult"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """MotionPlanResult[] steps
string path_to
bool success
int32 cond

================================================================================
MSG: tamp_ros/MotionPlanResult
std_msgs/Float32MultiArray[] traj
string failed
bool success
int32 plan_id
int32 cond
string task

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
  __slots__ = ['steps','path_to','success','cond']
  _slot_types = ['tamp_ros/MotionPlanResult[]','string','bool','int32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       steps,path_to,success,cond

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(HLPlanResult, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.steps is None:
        self.steps = []
      if self.path_to is None:
        self.path_to = ''
      if self.success is None:
        self.success = False
      if self.cond is None:
        self.cond = 0
    else:
      self.steps = []
      self.path_to = ''
      self.success = False
      self.cond = 0

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
      length = len(self.steps)
      buff.write(_struct_I.pack(length))
      for val1 in self.steps:
        length = len(val1.traj)
        buff.write(_struct_I.pack(length))
        for val2 in val1.traj:
          _v1 = val2.layout
          length = len(_v1.dim)
          buff.write(_struct_I.pack(length))
          for val4 in _v1.dim:
            _x = val4.label
            length = len(_x)
            if python3 or type(_x) == unicode:
              _x = _x.encode('utf-8')
              length = len(_x)
            buff.write(struct.pack('<I%ss'%length, length, _x))
            _x = val4
            buff.write(_get_struct_2I().pack(_x.size, _x.stride))
          buff.write(_get_struct_I().pack(_v1.data_offset))
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sf'%length
          buff.write(struct.pack(pattern, *val2.data))
        _x = val1.failed
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_get_struct_B2i().pack(_x.success, _x.plan_id, _x.cond))
        _x = val1.task
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.path_to
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_Bi().pack(_x.success, _x.cond))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.steps is None:
        self.steps = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.steps = []
      for i in range(0, length):
        val1 = tamp_ros.msg.MotionPlanResult()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.traj = []
        for i in range(0, length):
          val2 = std_msgs.msg.Float32MultiArray()
          _v2 = val2.layout
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          _v2.dim = []
          for i in range(0, length):
            val4 = std_msgs.msg.MultiArrayDimension()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            start = end
            end += length
            if python3:
              val4.label = str[start:end].decode('utf-8')
            else:
              val4.label = str[start:end]
            _x = val4
            start = end
            end += 8
            (_x.size, _x.stride,) = _get_struct_2I().unpack(str[start:end])
            _v2.dim.append(val4)
          start = end
          end += 4
          (_v2.data_offset,) = _get_struct_I().unpack(str[start:end])
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sf'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = struct.unpack(pattern, str[start:end])
          val1.traj.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.failed = str[start:end].decode('utf-8')
        else:
          val1.failed = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.success, _x.plan_id, _x.cond,) = _get_struct_B2i().unpack(str[start:end])
        val1.success = bool(val1.success)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.task = str[start:end].decode('utf-8')
        else:
          val1.task = str[start:end]
        self.steps.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.path_to = str[start:end].decode('utf-8')
      else:
        self.path_to = str[start:end]
      _x = self
      start = end
      end += 5
      (_x.success, _x.cond,) = _get_struct_Bi().unpack(str[start:end])
      self.success = bool(self.success)
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
      length = len(self.steps)
      buff.write(_struct_I.pack(length))
      for val1 in self.steps:
        length = len(val1.traj)
        buff.write(_struct_I.pack(length))
        for val2 in val1.traj:
          _v3 = val2.layout
          length = len(_v3.dim)
          buff.write(_struct_I.pack(length))
          for val4 in _v3.dim:
            _x = val4.label
            length = len(_x)
            if python3 or type(_x) == unicode:
              _x = _x.encode('utf-8')
              length = len(_x)
            buff.write(struct.pack('<I%ss'%length, length, _x))
            _x = val4
            buff.write(_get_struct_2I().pack(_x.size, _x.stride))
          buff.write(_get_struct_I().pack(_v3.data_offset))
          length = len(val2.data)
          buff.write(_struct_I.pack(length))
          pattern = '<%sf'%length
          buff.write(val2.data.tostring())
        _x = val1.failed
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
        _x = val1
        buff.write(_get_struct_B2i().pack(_x.success, _x.plan_id, _x.cond))
        _x = val1.task
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self.path_to
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_get_struct_Bi().pack(_x.success, _x.cond))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.steps is None:
        self.steps = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.steps = []
      for i in range(0, length):
        val1 = tamp_ros.msg.MotionPlanResult()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.traj = []
        for i in range(0, length):
          val2 = std_msgs.msg.Float32MultiArray()
          _v4 = val2.layout
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          _v4.dim = []
          for i in range(0, length):
            val4 = std_msgs.msg.MultiArrayDimension()
            start = end
            end += 4
            (length,) = _struct_I.unpack(str[start:end])
            start = end
            end += length
            if python3:
              val4.label = str[start:end].decode('utf-8')
            else:
              val4.label = str[start:end]
            _x = val4
            start = end
            end += 8
            (_x.size, _x.stride,) = _get_struct_2I().unpack(str[start:end])
            _v4.dim.append(val4)
          start = end
          end += 4
          (_v4.data_offset,) = _get_struct_I().unpack(str[start:end])
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          pattern = '<%sf'%length
          start = end
          end += struct.calcsize(pattern)
          val2.data = numpy.frombuffer(str[start:end], dtype=numpy.float32, count=length)
          val1.traj.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.failed = str[start:end].decode('utf-8')
        else:
          val1.failed = str[start:end]
        _x = val1
        start = end
        end += 9
        (_x.success, _x.plan_id, _x.cond,) = _get_struct_B2i().unpack(str[start:end])
        val1.success = bool(val1.success)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.task = str[start:end].decode('utf-8')
        else:
          val1.task = str[start:end]
        self.steps.append(val1)
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.path_to = str[start:end].decode('utf-8')
      else:
        self.path_to = str[start:end]
      _x = self
      start = end
      end += 5
      (_x.success, _x.cond,) = _get_struct_Bi().unpack(str[start:end])
      self.success = bool(self.success)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_B2i = None
def _get_struct_B2i():
    global _struct_B2i
    if _struct_B2i is None:
        _struct_B2i = struct.Struct("<B2i")
    return _struct_B2i
_struct_Bi = None
def _get_struct_Bi():
    global _struct_Bi
    if _struct_Bi is None:
        _struct_Bi = struct.Struct("<Bi")
    return _struct_Bi
