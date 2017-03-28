\version "2.10.33"

\header {
  title = "45 - Kommt her zu mir, spricht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 b4 fis'4 e4 
  fis8 e8 d4 cis4 b8 cis8 
  d4 cis4 d8 e8 fis4 
  e4 d4 cis4 d8 cis8 
  b4 b4 e4. d8   %5
  cis8 d16 e16 d8 cis8 b2 
  a4 a4 d4 cis4 
  d8 e8 fis4 e4 d4 
  cis4 b8 cis8 d4 cis4 
  d8 e8 fis4 e4 d4   %10
  cis4 fis4 fis4 fis4 
  e4. d8 cis4 d8 e8 
  cis2 b4
  
}

alto = \relative c' {
  \partial 4 fis4 
    g4 g4 fis8 gis8 ais8 b8 
  cis4 b4 ais4 b8 ais8 
  b4 fis4 fis4 fis4 
  b8 cis8 fis,8 gis8 ais4 b4 
  fis4 e8 fis8 gis2   %5
  a4 a4 a4 gis4 
  e4 fis4 b4 cis8 ais8 
  b4 b4 b8 cis8 fis,4 
  fis4 fis4 e4 e4 
  fis8 g8 a4 g4 fis4   %10
  fis4 fis8 e8 d8 fis8 b8 a8 
  g8 fis8 e2 d8 g8 
  g4 fis8 e8 dis4  
}

tenor = \relative c' {
  \partial 4 d4 
  e4 e4 d4 g!4 
  fis4 fis4 fis4 fis8 e8 
  fis4 fis8 e8 d8 cis8 b8 d8 
  d8 cis8 d8 b8 cis4 fis8 e8 
  d8 cis8 b4 b4 cis8 d8   %5
  e4 a,4 fis'4 e8 d8 
  cis4 d8 e8 fis4 fis4 
  fis8 e8 d4 d8 cis8 cis8 b8 
  ais4 d8 cis8 b8 gis8 a4 
  a4 d4 d8 cis8 cis8 b8   %10
  ais4 cis4 cis4 b4 
  b2 a8 g'8 fis8 b,8 
  b4 ais4 fis4
  
}

baixo = \relative c {
  \partial 4 b4 
  e8 fis8 g8 a8 b8 d8 cis8 b8 
  ais8 fis8 b4 fis4 d8 cis8 
  b8 b'8 ais8 fis8 b8 cis8 d8 b8 
  gis8 ais8 b8 b,8 fis'4 b,4 
  b'8 a8 gis8 fis8 e8 d8 cis8 b8   %5
  a8 cis8 fis8 e8 d8 b8 e4 
  a,4 d8 cis8 b8 b'8 ais8 fis8 
  b8 cis8 d8 b8 gis8 ais8 b8 b,8 
  fis'4 b8 a8 gis8 e8 a8 g8 
  fis8 e8 d8 cis8 b8 ais8 b4   %10
  fis'4 ais,4 b8 cis8 d8 dis8 
  e8 fis8 g8 gis8 a8 ais8 b8 g8 
  e8 cis8 fis4 b,4  
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