/**
 * @file class_macros.hpp
 * @brief This file contains some useful class macros.
 */

#ifndef OKVIS_CLASS_MACROS_HPP
#define OKVIS_CLASS_MACROS_HPP

#include <memory>

// These macros were inspired mainly on Maplab's macros
// https://github.com/ethz-asl/maplab

#define POINTER_TYPEDEFS(TypeName)                        \
  typedef std::shared_ptr<TypeName> Ptr;                  \
  typedef std::shared_ptr<const TypeName> ConstPtr;       \
  typedef std::unique_ptr<TypeName> UniquePtr;            \
  typedef std::unique_ptr<const TypeName> ConstUniquePtr; \
  typedef std::weak_ptr<TypeName> WeakPtr;                \
  typedef std::weak_ptr<const TypeName> WeakConstPtr;     \
  void definePointerTypedefs##__FILE__##__LINE__(void)

#define DELETE_COPY_CONSTRUCTORS(TypeName)        \
  TypeName(const TypeName&) = delete;             \
  void operator=(const TypeName&) = delete

#endif // OKVIS_CLASS_MACROS_HPP