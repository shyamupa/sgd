// -*- C++ -*-
// Little library of copy-on-write wrappers

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA



// $Id$


#ifndef WRAPPER_H
#define WRAPPER_H


template <class Rep> 
class Wrapper
{
 private:
  Rep *q;
  Rep *ref(Rep *q) { q->refcount++; return q; }
  void deref(Rep *q) { if (! --(q->refcount)) delete q; }
  
 public:
  Wrapper() 
    : q(new Rep) { q->refcount = 1; }
  Wrapper(const Wrapper<Rep> &other) 
    : q(ref(other.q)) {}
  ~Wrapper() 
    { deref(q); }
  Wrapper& operator=(const Wrapper<Rep> &other) 
    { if (&other != this) { deref(q); q=ref(other.q); } return *this; }
  void detach() 
    { if (q->refcount > 1) { deref(q); q=q->copy(); q->refcount=1; } }
  Rep *rep() const
    { return q; }
};



// Recommended usage
//
// #include <cstdlib>
// #include <cstring>
// 
// class String
// {
// private:
//
//   struct Rep
//   {
//     int refcount;
//     int length;
//     char *data;
//     Rep(const char *s, int l)
//       : length(len), data(new char[l+1]) 
//       { ::memcpy(data, s, l);  data[len] = 0; }
//     Rep *copy()
//       { return new StringRep(data, length); }
//   };
//
//   Wrapper<Rep> w;
//   Rep *rep() { return w.rep(); }
//   const Rep *rep() const { return w.rep(); }
//     
// public:
//   String(const char *s, int l)
//     : w(Rep(s,l)) {}
//   String(const char *s)
//     : w(Rep(s,::strlen(s))) {}
// 
//   // function that do not mutate
//   int size() const { return rep()->length; }
//   operator const char*() const { return rep()->data; }
//   char operator[](int i) const { return rep()->data[i]; }
//   
//   // functions that perform a mutation
//   void set(int i, char c) { detach(); rep()->data[i] = c; }
// }
// 



#endif



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" )
   End:
   ------------------------------------------------------------- */
