// General US-English numeral verbalization. No sample IDs or dataset-specific rules.
#pragma once
#include <regex>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdint>
namespace bryce_text {
inline std::string cardinal(uint64_t n){
 static const char* small[]={"zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"};
 static const char* tens[]={"","","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"};
 if(n<20)return small[n];if(n<100)return std::string(tens[n/10])+(n%10?" "+cardinal(n%10):"");
 for(auto entry:{std::pair<uint64_t,const char*>{1000000000,"billion"},{1000000,"million"},{1000,"thousand"},{100,"hundred"}})if(n>=entry.first)return cardinal(n/entry.first)+" "+entry.second+(n%entry.first?" "+cardinal(n%entry.first):"");
 return "";
}
inline std::string ordinal(uint64_t n){
 static const char* small[]={"zeroth","first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth"};
 if(n<20)return small[n];if(n<100){if(n%10)return cardinal(n/10*10)+" "+small[n%10];static const char* t[]={"","","twentieth","thirtieth","fortieth","fiftieth","sixtieth","seventieth","eightieth","ninetieth"};return t[n/10];}
 if(n%100)return cardinal(n/100*100)+" "+ordinal(n%100);return cardinal(n)+"th";
}
inline std::string year(uint64_t n){if(n>=2000&&n<2010)return "two thousand"+(n%100?" "+cardinal(n%100):"");return cardinal(n/100)+(n%100==0?" hundred":n%100<10?" oh "+cardinal(n%100):" "+cardinal(n%100));}
inline std::string NormalizeEnglishNumbers(const std::string&text){
 static const std::regex number("\\b[0-9][0-9,]*(st|nd|rd|th|s)?\\b");
 static const std::regex year_before("(in|since|during|throughout|from|until|before|after|between|year|fiscal)\\s*$",std::regex::icase);
 static const std::regex year_after("^\\s+(campaign|election|season|census|olympics|games)\\b",std::regex::icase);
 static const std::regex month_before("(january|february|march|april|may|june|july|august|september|october|november|december)\\s+(\\d+[, ]+)?$",std::regex::icase);
 std::string result;size_t last=0;
 for(std::sregex_iterator it(text.begin(),text.end(),number),end;it!=end;++it){const auto&m=*it;size_t pos=m.position(),count=m.length();result+=text.substr(last,pos-last);std::string token=m.str(),suffix=m[1].str();
  bool decimal=(pos>1&&text[pos-1]=='.'&&std::isdigit(static_cast<unsigned char>(text[pos-2])))||(pos+count+1<text.size()&&text[pos+count]=='.'&&std::isdigit(static_cast<unsigned char>(text[pos+count+1])));
  std::string digits=token.substr(0,token.size()-suffix.size());digits.erase(std::remove(digits.begin(),digits.end(),','),digits.end());
  if(decimal||digits.size()>10){result+=token;last=pos+count;continue;}
  uint64_t value=std::stoull(digits);std::string before=text.substr(pos>40?pos-40:0,std::min<size_t>(40,pos)),after=text.substr(pos+count,30);
  bool is_year=digits.size()==4&&value>=1000&&value<=2099&&(std::regex_search(before,year_before)||std::regex_search(before,month_before)||std::regex_search(after,year_after)||suffix=="s");
  if(suffix=="s"&&is_year&&value%10==0){static const char* decades[]={"hundreds","tens","twenties","thirties","forties","fifties","sixties","seventies","eighties","nineties"};result+=cardinal(value/100)+" "+decades[value%100/10];}
  else if(!suffix.empty()&&suffix!="s")result+=ordinal(value);
  else result+=is_year?year(value):cardinal(value);
  if(suffix=="s"&&!is_year)result+="s";
  last=pos+count;
 }
 result+=text.substr(last);return result;
}
}
