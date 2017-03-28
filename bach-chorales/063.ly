\version "2.10.33"

\header {
  title = "63 - Nun ruhen alle Wälder"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 cis4 
    a4 b4 cis8 d8 e4 
  d2 cis4 cis4 
  e4 e4 b4 d8 cis8 
  a2 gis4 e4 
  a4 b4 cis8 d16 e16 d8 cis8   %5
  b2. cis4 
  a4 b4 cis8 d8 e4 
  d2 cis4 cis8 d8 
  e4 e4 b4 cis4 
  a2 gis4 e4   %10
  a4 b4 cis8 d8 e8 d8 
  cis4 b4 a
  
}

alto = \relative c'' {
  \partial 4 a4 
    fis4 gis4 a8 b8 cis4 
  cis4 b8 a8 gis4 a4 
  a4 a4 gis4 b8 a8 
  fis2 e4 b4 
  e4 fis8 gis8 a8 b16 cis16 b8 a8   %5
  gis2. gis4 
  fis4 e4 e4 fis4 
  fis4 eis4 fis4 fis4 
  e4 a8 gis8 fis4 e8 dis16 cis16 
  fis2 e4 e4   %10
  e4 d4 cis8 e8 a4 
  a4 gis4 e 
  
}

tenor = \relative c' {
  \partial 4 e4 
    d4 d4 e4 a,4 
  a4 b4 cis4 fis4 
  e4 e4 e4 e4 
  e4 dis4 gis,4 gis4 
  a4 d4 e8 a,8 gis8 a8   %5
  e'2. cis4 
  cis4 b4 a8 b8 cis4 
  b4 gis4 ais4 a4 
  a4 a4 a4 gis4 
  cis4 b4 b4 cis4   %10
  cis4 b4 e4. fis8 
  e4 d4 cis 
  
}

baixo = \relative c' {
  \partial 4 a4 
    d4 cis8 b8 a4. g8 
  fis2 eis4 fis4 
  cis8 b8 cis8 d8 e4 a,4 
  b4 bis4 cis4 d4 
  cis4 b4 a4 d4   %5
  e2. eis4 
  fis4 gis4 a4 ais4 
  b2 fis4 fis4 
  cis8 b8 cis4 dis4 e4 
  e4 dis4 e4 cis4   %10
  fis8 e8 fis8 gis8 a4 cis,8 d8 
  e4 e4 a, 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}