\version "2.10.33"

\header {
  title = "280 - Eins ist not! ach Herr, dies Eine"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  a4 g8 fis8 b4 b4 
  cis4 cis4 d4 d4 
  b4. a8 g4 a4 
  g8 fis8 e8 fis8 d2 
  a'4 g8 fis8 b4 b4   %5
  cis4 cis4 d4 d4 
  b4. a8 g4 a4 
  g8 fis8 e8 fis8 d2
  \time 3/4
  \partial 4 d8 e8
  fis4 fis4 fis4 
  e4 e4 e8 fis8
  g4 g4 g8 fis8
  fis4 fis4 fis4
  b4. a8 g4 
  a4. g8 fis4
  g4 fis4 e4
  d4 d4 d8 e8
  fis4 fis4 fis4
  e4 e4 e8 fis8
  g4 g4 g4
  fis2 fis4
  b4. a8 g4 
  a4. g8 fis4
  g4 fis4 e4
  d2.
}

alto = \relative c' {
  fis4 d4 d4 g8 fis8 
  e4 a8 g8 fis4 fis4 
  d4 g8 fis8 e4 e4 
  d4 cis4 a2 
  e'4 d4 fis4 e8 d8
  cis4 fis4 fis4 fis4 
  g8 fis8 e4 d4 e4 
  e8 d8 d8 cis8 a2 
  \time 3/4
  \partial 4 a4
  d4 d4 d4 
  cis8 b8 cis8 d8 e4
  e4 e4 e4
  d8 cis8 d8 e8 fis4
  g4. fis8 e4 
  e8 d8 cis4 d4
  d4 d4 cis4
  a4 a4 a4
  d4 d4 d4
  d4 cis8 b8 cis8 d8
  e4 e4 e4
  e4 d4 r4
  d4 g8 fis8 e4~ 
  e8 g8 fis8 e8 d8 a8
  d4. e8 cis4
  a2.
}

tenor = \relative c' {
  d4 a4 b4 e8 d8 
  cis8 b8 a4 a4 a4 
  g8 a8 b4 b4 a4 
  a4 a8 g8 fis2 
  a4 a4 a8 fis8 g8 b8~
  b8 gis8 ais4 b4 b4 
  d4 cis4 b8 g8 e8 a8 
  a4 a4 fis2
  \time 3/4
  \partial 4 fis8 g8
  a4 a4 a8 b8 
  cis8 d8 e8 d8 cis4
  b4 b4 b8 cis8
  d8 e8 fis8 e8 d8 cis8
  b8 a8 g8 a8 b4 
  a4 a4 a4
  g8 a8 b4 e,4
  fis4 fis4 fis4
  b4 b4 a4
  b4 a8 gis8 a4
  a4 b4 cis4
  a2 a4
  d8 cis8 b4. b8 
  a4. cis8 d4
  d,4 a'4 a8 g8
  fis2.
}

baixo = \relative c {
  d4 d4 g8 fis8 e4 
  a8 g8 fis8 e8 d4 d4 
  g8 fis8 e4 e8 d8 cis4 
  d4 a4 d2 
  cis4 d4 dis4 e4
  f4 fis4 b,4 b4 
  g4 a4 b4 cis4 
  d4 a4 d2 
  \time 3/4
  \partial 4 d4
  d8 cis8 d8 e8 fis8 g8 
  a4 a,4 a'4
  e8 dis8 e8 fis8 g8 a8
  b4 b,4 b'8 a8
  g8 fis8 e4 e8 d8 
  cis8 b8 a4 d8 cis8
  b8 a8 g4 a4
  d4 d4 d8 cis8
  b4 b'8 a8 g8 fis8
  gis4 e4 a4
  cis,4 d8 cis8 b8 a8
  d2 r4
  b4 e4 e8 d8 
  cis8 a8 d4. cis8
  b4 a8 g8 a4
  d2.
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