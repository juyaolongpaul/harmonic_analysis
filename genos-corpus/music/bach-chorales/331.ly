\version "2.10.33"

\header {
  title = "331 - Wo soll ich fliehen hin"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key b \minor
}


soprano = \relative c'' {
  \partial 4 b4 
    b4 cis4 d4 e4 
  fis2. fis4 
  fis4 e4 d4 e4 
  cis2. cis4 
  d4 e4 fis4 fis4   %5
  e2 fis4 fis4 
  d4 e4 fis4 fis4 
  e2 d4 fis4 
  a4 fis4 fis4 fis4 
  e2 e4 e4   %10
  fis4 e4 d4 e8 d8 
  cis2 b4 
}

alto = \relative c' {
  \partial 4 fis4 
    g4 fis4 fis4 g4 
  cis,2. fis4 
  gis4 ais4 b4 cis4 
  ais2. ais4 
  b4 e,4 d4 a'4   %5
  a2 ais4 fis4 
  fis8 gis8 a4 a4 a4 
  b4 a4 a4 a4 
  a4 d4 cis4 b4 
  b2 e,4 b'4   %10
  cis4 cis4 fis,4 g4 
  g4 fis4 fis4 
}

tenor = \relative c' {
  \partial 4 d4 
    e4 e4 d4 cis8 b8 
  ais2. ais4 
  b4 cis4 d4 g4 
  fis2. fis,4 
  fis4 a4 a4. b8   %5
  cis2 cis4 cis4 
  b4 cis4 d4 d4 
  d4 cis4 d4 d4 
  e4 a,8 d8 e8 cis8 d4 
  d2 cis4 b4   %10
  fis'4 ais,4 b4 b4 
  e8 d8 cis4 dis4 
}

baixo = \relative c' {
  \partial 4 b4 
    e4 ais,4 b8 a8 g4 
  fis2. e4 
  d4 cis4 b4 e4 
  fis2. fis4 
  b,4 cis4 d8 e8 fis8 g8   %5
  a8 b8 a8 g8 fis8 gis8 ais8 fis8 
  b4 a8 g8 fis8 e8 d8 fis8 
  g8 e8 a8 a,8 d4 d'4 
  cis4 d4 ais4 b8 a8 
  gis8 fis8 gis8 e8 a4 g4   %10
  fis8 e8 d8 cis8 b'8 a8 g8 fis8 
  e8 cis8 fis8 fis,8 b4 
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