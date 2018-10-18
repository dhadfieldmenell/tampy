// Generated by gencpp from file tamp_ros/PrimitiveRequest.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_PRIMITIVEREQUEST_H
#define TAMP_ROS_MESSAGE_PRIMITIVEREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace tamp_ros
{
template <class ContainerAllocator>
struct PrimitiveRequest_
{
  typedef PrimitiveRequest_<ContainerAllocator> Type;

  PrimitiveRequest_()
    : prim_obs()  {
    }
  PrimitiveRequest_(const ContainerAllocator& _alloc)
    : prim_obs(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename ContainerAllocator::template rebind<float>::other >  _prim_obs_type;
  _prim_obs_type prim_obs;





  typedef boost::shared_ptr< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> const> ConstPtr;

}; // struct PrimitiveRequest_

typedef ::tamp_ros::PrimitiveRequest_<std::allocator<void> > PrimitiveRequest;

typedef boost::shared_ptr< ::tamp_ros::PrimitiveRequest > PrimitiveRequestPtr;
typedef boost::shared_ptr< ::tamp_ros::PrimitiveRequest const> PrimitiveRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tamp_ros::PrimitiveRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace tamp_ros

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'tamp_ros': ['/home/michaelmcdonald/dependencies/tampy/src/ros_utils/src/tamp_ros/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "54454e5de4cab21b3253a7303e1de1f3";
  }

  static const char* value(const ::tamp_ros::PrimitiveRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x54454e5de4cab21bULL;
  static const uint64_t static_value2 = 0x3253a7303e1de1f3ULL;
};

template<class ContainerAllocator>
struct DataType< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tamp_ros/PrimitiveRequest";
  }

  static const char* value(const ::tamp_ros::PrimitiveRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] prim_obs\n\
\n\
";
  }

  static const char* value(const ::tamp_ros::PrimitiveRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.prim_obs);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PrimitiveRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tamp_ros::PrimitiveRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tamp_ros::PrimitiveRequest_<ContainerAllocator>& v)
  {
    s << indent << "prim_obs[]" << std::endl;
    for (size_t i = 0; i < v.prim_obs.size(); ++i)
    {
      s << indent << "  prim_obs[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.prim_obs[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAMP_ROS_MESSAGE_PRIMITIVEREQUEST_H