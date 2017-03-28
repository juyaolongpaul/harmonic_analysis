\version "2.10.33"

\header {
  title = "342 - Lobt Gott, ihr Christen, allzugleich"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    e'4 e4 e4 e4 
  fis4 e8 d8 cis4 b4 
  cis8 dis8 e4 e8 fis8 dis4 
  e2. e4 
  e4 e4 e8 d8 cis8 d8   %5
  e4 d8 cis8 b4 e4 
  d4 cis4 b4. cis8 
  a8 b8 cis8 d8 e4 e4 
  d4 cis4 b4 b4 
  a2.   %10
  
}

alto = \relative c' {
  \partial 4 e4 
    e8 fis8 gis4 a4 a4 
  a4 gis4 e4 e4 
  e8 fis8 gis4 fis4 gis8 a8 
  gis2. gis4 
  a4 gis8 fis8 e4 e8 d8   %5
  cis8 a8 fis'4 gis4 cis8 b8 
  a4. gis8 fis4 gis4 
  fis8 gis8 a8 b8 cis4 fis,4 
  fis8 gis8 a4 a4 gis4 
  e2.   %10
  
}

tenor = \relative c' {
  \partial 4 cis4 
    b4 e8 d8 cis8 d8 e4 
  d8 cis8 b4 a4 gis4 
  a4 b4 b4 b4 
  b2. b4 
  cis4 b8 a8 b4 a4   %5
  a4 a4 e'4 gis,4 
  a8 b8 cis8 a8 d4. cis8 
  cis4 fis,4 gis4 cis4 
  d8 e8 fis4 fis4 e8 d8 
  cis2.   %10
  
}

baixo = \relative c' {
  \partial 4 a4 
    gis4 e4 a8 b8 cis8 a8 
  d,4 e4 a,4 e'4 
  a4 gis8 a8 b4 b,4 
  e2. e8 d8 
  cis8 d8 e8 fis8 gis8 e8 a4   %5
  cis,4 d4 e4 cis4 
  fis8 gis8 a4. gis8 fis8 eis8 
  fis4 e8 d8 cis4 ais'4 
  b4 fis8 e8 d8 b8 e4 
  a,2.   %10
  
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