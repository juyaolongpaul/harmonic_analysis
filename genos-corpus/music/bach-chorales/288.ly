\version "2.10.33"

\header {
  title = "288 - Gelobet seist du, Jesu Christ"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 b4 a4 
  d4 e4 d2 
  cis8 d8 e4 fis4 e8 d8 
  cis4 b4 a4 a4 
  d4 cis8 d8 e4 b4   %5
  a8 gis8 fis4 e4 e4 
  b'4 b4 b4 cis4 
  d8 cis8 b4 a8 b8 cis4 
  b2 a4 a4 
  a2 a4   %10
  
}

alto = \relative c' {
  \partial 4 cis8  d8 
    e4 fis4 fis8 e8 e4 
  a4 a8 g8 fis2 
  e8 fis8 g4 a8 fis8 b4 
  b8 a8 a8 gis8 e4 fis4 
  fis8 g8 a4 a4 gis4   %5
  fis8 e8 e8 dis8 b4 cis4 
  dis4 dis4 e4 e4 
  d4. e8 fis4 e8 fis8 
  gis2 fis4 e4~ 
  e8 d16 e16 fis4 e4   %10
  
}

tenor = \relative c' {
  \partial 4 a4 
    a4 d4 d4 cis4 
  d8 e16 fis16 cis4 a2 
  a4 b4 cis8 d8 e4 
  e4 e4 cis4 d4 
  d8 e8 fis4 e4 e8 b8   %5
  b4 b4 gis4 a8 gis8 
  fis4 b8 a8 gis4 ais4 
  b8 a8 g4 fis8 gis8 a4 
  e'2 cis8 d8 e8 g,8 
  fis8 a8 d4 cis4   %10
  
}

baixo = \relative c {
  \partial 4 fis4 
    cis4 d4 g4. fis16 e16 
  fis8 d8 a'4 d,2 
  a'4 g4 fis4 gis4 
  a4 e4 a,4 d8 cis8 
  b4 fis'4 cis8 d8 e4   %5
  dis8 e8 b4 e4 a4 
  a4 gis8 fis8 e8 d8 cis4 
  b4. cis8 d4 cis8 dis8 
  e2 fis4 cis4 
  d2 a4   %10
  
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