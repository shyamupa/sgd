

// -*- C++ -*-
// Little library of vectors and sparse vectors

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


#include "vectors.h"
#include <cassert>
#include <cctype>


namespace
{
  template <typename T> inline T min(T a, T b) {
    return (a < b) ? a  : b;
  }
  template <typename T> inline T max(T a, T b) {
    return (a < b) ? b  : a;
  }
}



void 
FVector::Rep::resize(int n)
{
  if (size != n)
    {
      if (n > 0)
        {
          float *newdata = new float[n];
          int i = 0;
          int s = min(size,n);
          for (; i<s; i++)
            newdata[i] = data[i];
          for (; i<n; i++)
            newdata[i] = 0;
          delete [] data;
          data = newdata;
          size = n;
        }
      else
        {
          delete [] data;
          data = 0;
          size = 0;
        }
    }
}


FVector::Rep *
FVector::Rep::copy()
{
  Rep *newrep = new Rep;
  newrep->resize(size);
  for (int i=0; i<size; i++)
    newrep->data[i] = data[i];
  return newrep;
}


FVector::FVector()
{
}


FVector::FVector(int n)
{
  Rep *r = rep();
  r->resize(n);
}


FVector::FVector(const SVector &v)
{
  Rep *r = rep();
  r->resize(v.size());
  int npairs = v.npairs();
  const SVector::Pair *pairs = v;
  for (int i=0; i<npairs; i++, pairs++)
    r->data[pairs->i] = pairs->v;
}


double
FVector::get(int i) const
{
  const Rep *r = rep();
  if (i >=0 && i<r->size)
    return r->data[i];
  assert(i>=0);
  return 0;
}


double 
FVector::set(int i, double v)
{
  w.detach();
  Rep *r = rep();
  if (i >= r->size)
    r->resize(i+1);
  assert(i>=0);
  r->data[i] = v;
  return v;
}


void 
FVector::clear()
{
  w.detach();
  rep()->resize(0);
}


void 
FVector::resize(int n)
{
  w.detach();
  assert(n >= 0);
  rep()->resize(n);
}


void 
FVector::add(double c1)
{
  w.detach();
  Rep *r = rep();
  float *d = r->data;
  for (int i=0; i<r->size; i++)
    d[i] += c1;
}


void 
FVector::add(const FVector &v2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] += s[i];
}


void 
FVector::add(const SVector &v2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    d[pairs->i] += pairs->v;
}


void 
FVector::add(const FVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] += s[i] * c2;
}


void 
FVector::add(const SVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs; i++, pairs++)
    d[pairs->i] += pairs->v * c2;
}


void 
FVector::add(const FVector &v2, double c2, const FVector &q2)
{
  w.detach();
  Rep *r = rep();
  int m = r->size;
  m = max(m, v2.size());
  m = min(m, q2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  const float *q = (const float*) q2;
  for (int i=0; i<m; i++)
    d[i] += s[i] * q[i] * c2;
}


void 
FVector::add(const SVector &v2, double c2, const FVector &q2)
{
  w.detach();
  Rep *r = rep();
  int m = r->size;
  m = max(m, v2.size());
  m = min(m, q2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *q = (const float*) q2;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  for (int i=0; i<npairs && pairs->i < m; i++, pairs++)
    d[pairs->i] += pairs->v * q[pairs->i] * c2;
}


void 
FVector::scale(double c1)
{
  w.detach();
  Rep *r = rep();
  float *d = r->data;
  for (int i=0; i<r->size; i++)
    d[i] *= c1;
}


void 
FVector::combine(double c1, const FVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  const float *s = (const float*) v2;
  for (int i=0; i<m; i++)
    d[i] = d[i] * c1 + s[i] * c2;
}


void 
FVector::combine(double c1, const SVector &v2, double c2)
{
  w.detach();
  Rep *r = rep();
  int m = max(r->size, v2.size());
  if (m > r->size)
    r->resize(m);
  float *d = r->data;
  int npairs = v2.npairs();
  const SVector::Pair *pairs = v2;
  int j = 0;
  for (int i=0; i<npairs; i++, pairs++)
    {
      if (c1 != 1)
        for (; j < pairs->i; j++)
          d[j] = d[j] * c1;
      j = pairs->i;
      d[j] = d[j] * c1 + pairs->v * c2;
      j++;
    }
  if (c1 != 1)
    for (; j < pairs->i; j++)
      d[j] = d[j] * c1;
}




std::ostream& 
operator<<(std::ostream &f, const FVector &v)
{
  SVector s(v);
  f << s;
  return f;
}


std::istream& 
operator>>(std::istream &f, FVector &v)
{
  SVector s;
  f >> s;
  if (f.good())
    v = s;
  return f;
}

bool 
FVector::save(std::ostream &f) const
{
  int i = size();
  const float *d = rep()->data;
  f.write((const char*)&i, sizeof(int));
  f.write((const char*)d, sizeof(float)*i);
  return f.good();
}


bool
FVector::load(std::istream &f)
{
  int i = 0;
  clear();
  f.read((char*)&i, sizeof(int));
  if (i<0)
    f.setstate(std::ios::badbit);
  if (!f.good())
    return false;
  resize(i);
  float *d = rep()->data;
  f.read((char*)d, sizeof(float)*i);
  return f.good();
}



// ----------------------------------------


void
SVector::Rep::resize(int n)
{
  if (n != mpairs)
    {
      Pair *p = new Pair[n+1];
      int m = min(n, npairs);
      int i = 0;
      for (; i < m; i++)
        p[i] = pairs[i];
      for (; i <= n; i++)
        p[i].i = -1;
      delete [] pairs;
      pairs = p;
      npairs = m;
      mpairs = n;
      size = (m>0) ? p[m-1].i + 1 : 0;
    }
}


SVector::Rep *
SVector::Rep::copy()
{
  int n = npairs;
  Pair *p = new Pair[n+1];
  for (int i=0; i < n; i++)
    p[i] = pairs[i];
  p[n].i = -1;
  Rep *nr = new Rep;
  delete [] nr->pairs;
  nr->pairs = p;
  nr->size = size;
  nr->npairs = nr->mpairs = n;
  return nr;
}


inline double
SVector::Rep::qset(int i, double v)
{
  assert(i >= size);
  if (npairs >= mpairs)
    resize(npairs + min(16, max(mpairs, 4096)));
  Pair *p = &pairs[npairs++];
  size = i + 1;
  p->i = i;
  p->v = v;
  return v;
}


SVector::SVector()
{
  trim();
}


SVector::SVector(const FVector &v)
{
  int m = v.size();
  const float *f = v;
  Rep *r = rep();
  r->resize(m);
  for (int i=0; i<m; i++)
    if (f[i] != 0)
      r->qset(i,f[i]);
  trim();
}


namespace {
  
  SVector::Pair *
  search(SVector::Pair *pairs, int npairs, int i)
  {
    int lo = 0;
    int hi = npairs - 1;
    while (lo <= hi)
      {
        int d = (lo + hi + 1) / 2;
        if (i == pairs[d].i)
          return &pairs[d];
        else if (i < pairs[d].i)
          hi = d-1;
        else
          lo = d+1;
      }
    return 0;
  }

}


double
SVector::get(int i) const
{
  const Rep *r = rep();
  if (i < 0 || i >= r->size)
    return 0;
  // binary search
  SVector::Pair *pair = search(r->pairs, r->npairs, i);
  if (pair)
    return pair->v;
  return 0;
}


double 
SVector::set(int i, double v)
{
  w.detach();
  Rep *r = rep();
  if (v)
    {
      if (i >= r->size)
        return r->qset(i, v);
      SVector::Pair *p = search(r->pairs, r->npairs, i);
      if (p)
        return p->v = v;
      if (r->npairs >= r->mpairs)
        r->resize(r->npairs + min(16, max(r->mpairs, 4096)));
      SVector::Pair *s = r->pairs;
      p = s + r->npairs;
      r->npairs += 1;
      for (; p > s && p[-1].i > i;  p--)
        p[0] = p[-1];
      p[0].i = i;
      p[0].v = v;
    }
  else
    {
      SVector::Pair *s = r->pairs;
      SVector::Pair *p = search(s, r->npairs, i);
      if (p)
        {
          r->npairs -= 1;
          for (; p->i >= 0; p++)
            p[0] = p[1];
        }
    }
  return v;
}


void 
SVector::clear()
{
  w.detach();
  rep()->resize(0);
}

void 
SVector::trim()
{
  w.detach();
  Rep *r = rep();
  r->resize(r->npairs);
}


void 
SVector::add(const SVector &v2)
{
  operator=( ::combine(*this, 1, v2, 1) );
}


void 
SVector::add(const SVector &v2, double c2)
{
  operator=( ::combine(*this, 1, v2, c2) );
}


void 
SVector::combine(double c1, const SVector &v2, double c2)
{
  operator=( ::combine(*this, c1, v2, c2) );
}


void 
SVector::scale(double c1)
{
  if (c1)
    {
      w.detach();
      Rep *r = rep();
      Pair *pairs = r->pairs;
      int npairs = r->npairs;
      for (int i=0; i<npairs; i++)
        pairs[i].v *= c1;
    }
  else
    {
      clear();
    }
}


std::ostream& 
operator<<(std::ostream &f, const SVector &v)
{
  const SVector::Rep *r = v.rep();
  const SVector::Pair *pairs = r->pairs;
  int npairs = r->npairs;
  for (int i=0; i<npairs; i++)
    f << " " << pairs[i].i << ":" << pairs[i].v;
  f << std::endl;
  return f;
}


std::istream& 
operator>>(std::istream &f, SVector &v)
{
  v.clear();
  for(;;)
    {
      int c = f.get();
      if (!f.good() || (c=='\n' || c=='\r'))
        break;
      if (::isspace(c))
        continue;
      int i;
      f.unget();
      f >> std::skipws >> i >> std::ws;
      if (f.get() != ':')
        {
          f.unget();
          f.setstate(std::ios::badbit);
          break;
        }
      double x;
      f >> std::skipws >> x;
      if (!f.good())
        break;
      v.set(i,x);
    }
  v.trim();
  return f;
}


bool 
SVector::save(std::ostream &f) const
{
  const Rep *r = rep();
  const Pair *pairs = r->pairs;
  int npairs = r->npairs;
  f.write((const char*)&npairs, sizeof(int));
  f.write((const char*)pairs, sizeof(Pair)*npairs);
  return f.good();
}


bool 
SVector::load(std::istream &f)
{
  clear();
  int npairs = 0;
  f.read((char*)&npairs, sizeof(int));
  if (npairs < 0)
    f.setstate(std::ios::badbit);
  if (!f.good())
    return false;
  rep()->resize(npairs);
  for (int i=0; i<npairs; i++)
    {
      Pair pair;
      f.read((char*)&pair, sizeof(Pair));
      if (f.good())
        set(pair.i, pair.v);
    }
  trim();
  return f.good();
}


double 
dot(const FVector &v1, const FVector &v2)
{
  int m = min(v1.size(), v2.size());
  const float *f1 = v1;
  const float *f2 = v2;
  double sum = 0.0;
  while (--m >= 0)
    sum += (*f1++) * (*f2++);
  return sum;
}


double 
dot(const FVector &v1, const SVector &v2)
{
  int m = v1.size();
  const float *f = v1;
  const SVector::Pair *p = v2;
  double sum = 0;
  if (p)
    for( ; p->i >= 0 && p->i < m; p++)
      sum += p->v * f[p->i];
  return sum;
}


double 
dot(const SVector &v1, const FVector &v2)
{
  int m = v2.size();
  const float *f = v2;
  int n = v1.npairs();
  const SVector::Pair *p = v1;
  double sum = 0;
  if (p)
    for( ; p->i >= 0 && p->i < m; p++)
      sum += p->v * f[p->i];
  

  for( ; --n >= 0 && p->i < m; p++)
    sum += p->v * f[p->i];
  return sum;
}


double 
dot(const SVector &v1, const SVector &v2)
{
  const SVector::Pair *p1 = v1;
  const SVector::Pair *p2 = v2;
  double sum = 0;
  if (p1 && p2)
    while (p1->i >= 0 && p2->i >= 0)
      {
        if (p1->i < p2->i)
          p1++;
        else if (p1->i > p2->i)
          p2++;
        else
          sum += (p1++)->v * (p2++)->v;
      }
  return sum;
}


SVector 
combine(const SVector &v1, double a1, const SVector &v2, double a2)
{
  const SVector::Pair *p1 = v1;
  const SVector::Pair *p2 = v2;
  SVector ans;
  SVector::Rep *r = ans.rep();
  r->resize(v1.npairs() + v2.npairs());
  SVector::Pair *p = r->pairs;
  while (p1->i >= 0 && p2->i >= 0)
     {
       if (p1->i < p2->i)
         {
           double v = p1->v * a1;
           if (v)
             {
               p->i = p1->i;
               p->v = v;
               p++;
             }
           p1++;
         }
       else if (p1->i > p2->i)
         {
           double v = p2->v * a2;
           if (v)
             {
               p->i = p2->i;
               p->v = v;
               p++;
             }
           p2++;
         }
       else
         {
           double v = p1->v * a1 + p2->v * a2;
           if (v)
             {
               p->i = p1->i;
               p->v = v;
               p++;
             }
           p1++;
           p2++;
         }
     }
  while  (p1->i >= 0)
    {
      double v = p1->v * a1;
      if (v)
        {
          p->i = p1->i;
          p->v = v;
          p++;
        }
      p1++;
    }
  while  (p2->i >= 0)
    {
      double v = p2->v * a2;
      if (v)
        {
          p->i = p2->i;
          p->v = v;
          p++;
        }
      p2++;
    }
  r->npairs = p - r->pairs;
  r->size = (r->npairs > 0) ? p[-1].i + 1 : 0;
  ans.trim();
  return ans;
}


FVector 
combine(const FVector &v1, double a1, const SVector &v2, double a2)
{
  FVector r = v1;
  r.combine(a1, v2, a2);
  return r;
}


FVector 
combine(const SVector &v1, double a1, const FVector &v2, double a2)
{
  FVector r = v2;
  r.combine(a2, v1, a1);
  return r;
}


FVector 
combine(const FVector &v1, double a1, const FVector &v2, double a2)
{
  FVector r = v1;
  r.combine(a1, v2, a2);
  return r;
}



/* -------------------------------------------------------------
   Local Variables:
   c++-font-lock-extra-types: ( "\\sw+_t" "[A-Z]\\sw*[a-z]\\sw*" "std::\\sw+")
   End:
   ------------------------------------------------------------- */
