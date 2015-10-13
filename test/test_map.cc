#include <map>
#include <algorithm>
#include <string>
#include <iostream>
#include <cstring>
#include <cctype>

struct CaseInsenstiveComparator
{
  bool operator()(const std::string& a, const std::string& b) const
  {
    return strncasecmp(a.c_str(), b.c_str(), std::min(a.size(), b.size())) < 0;
  }
}; // CaseInsenstiveComparator

int main()
{
  std::map<std::string, std::string, CaseInsenstiveComparator> m;

  m["test"] = "world";
  m["Test"] = "hello";

  std::cout << m["test"] << std::endl;
  std::cout << m["Test"] << std::endl;
  return 0;
}

