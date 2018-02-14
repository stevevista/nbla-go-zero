// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/generated_message_util.h>

#include <limits>
#include <vector>

#include <google/protobuf/io/coded_stream_inl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/message_lite.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/wire_format_lite_inl.h>

namespace google {

namespace protobuf {
namespace internal {


double Infinity() {
  return std::numeric_limits<double>::infinity();
}
double NaN() {
  return std::numeric_limits<double>::quiet_NaN();
}

ExplicitlyConstructed< ::std::string> fixed_address_empty_string;
GOOGLE_PROTOBUF_DECLARE_ONCE(empty_string_once_init_);

void DeleteEmptyString() { fixed_address_empty_string.Destruct(); }

void InitEmptyString() {
  fixed_address_empty_string.DefaultConstruct();
  OnShutdown(&DeleteEmptyString);
}

size_t StringSpaceUsedExcludingSelfLong(const string& str) {
  const void* start = &str;
  const void* end = &str + 1;
  if (start <= str.data() && str.data() < end) {
    // The string's data is stored inside the string object itself.
    return 0;
  } else {
    return str.capacity();
  }
}



void InitProtobufDefaults() {
  GetEmptyString();
}

template <typename T>
const T& Get(const void* ptr) {
  return *static_cast<const T*>(ptr);
}


// We need to use a helper class to get access to the private members
class AccessorHelper {
 public:
  static int Size(const RepeatedPtrFieldBase& x) { return x.size(); }
  static void const* Get(const RepeatedPtrFieldBase& x, int idx) {
    return x.raw_data()[idx];
  }
};


void SerializeNotImplemented(int field) {
  GOOGLE_LOG(FATAL) << "Not implemented field number " << field;
}

// When switching to c++11 we should make these constexpr functions
#define SERIALIZE_TABLE_OP(type, type_class) \
  ((type - 1) + static_cast<int>(type_class) * FieldMetadata::kNumTypes)

int FieldMetadata::CalculateType(int type,
                                 FieldMetadata::FieldTypeClass type_class) {
  return SERIALIZE_TABLE_OP(type, type_class);
}


MessageLite* DuplicateIfNonNullInternal(MessageLite* message, Arena* arena) {
  if (message) {
    MessageLite* ret = message->New(arena);
    ret->CheckTypeAndMergeFrom(*message);
    return ret;
  } else {
    return NULL;
  }
}

// Returns a message owned by this Arena.  This may require Own()ing or
// duplicating the message.
MessageLite* GetOwnedMessageInternal(Arena* message_arena,
                                     MessageLite* submessage,
                                     Arena* submessage_arena) {
  GOOGLE_DCHECK(submessage->GetArena() == submessage_arena);
  GOOGLE_DCHECK(message_arena != submessage_arena);
  if (message_arena != NULL && submessage_arena == NULL) {
    message_arena->Own(submessage);
    return submessage;
  } else {
    MessageLite* ret = submessage->New(message_arena);
    ret->CheckTypeAndMergeFrom(*submessage);
    return ret;
  }
}

}  // namespace internal
}  // namespace protobuf
}  // namespace google
