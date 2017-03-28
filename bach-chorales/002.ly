\version "2.10.33"

\header {
  title = "2 - Ich dank' dir, lieber Herre"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 a4 a4 b4 
    g4 fis4 e4 b'4 
    cis4 b4 a4 gis8 fis8 
    gis4 fis4 e4 } e'4 
  d4 cis4 b4 a4   %5
  a8 b8 cis4 b4 cis4 
  d4 cis4 b4 ais4 
  b2. e,4 
  a4 b4 cis4 d4 
  e4 d8 cis8 b4 d4   %10
  cis4 b4 e4. d8 
  cis8 b8 a8 b8 cis4 b4 
  a2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    fis4 e4 fis4 fis4 
    e4 dis4 b4 gis'4 
    a4 gis8 fis8 e2~ 
    e4 dis4 b4 } cis'8 b8 
  a4 a4 gis4 a8 g8   %5
  fis8 gis8 a4 gis4 ais4 
  b8 a8 g4 fis8 e8 fis4 
  fis2. e8 d8 
  cis4 d4 e4 fis8 gis8 
  a2 e4 b'4   %10
  a8 gis8 fis4 e4 fis8 gis8 
  a2 a4 gis4 
  e2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 cis4 
    cis4 cis8 b8 a8 gis8 fis4 
    b4. a8 gis4 e'4 
    e4 dis4 cis2 
    b4. a8 gis4 } gis4 
  a8 b8 cis8 d8 e4 e4   %5
  d4 e4 e4 e4 
  fis8 b,4 ais8 b4 cis4 
  d2. gis,4 
  a4 gis8 fis8 e8 e'8 d4 
  cis8 d8 e8 fis8 gis4 fis4   %10
  fis8 e8 d8 cis8 b4 cis8 d8 
  e4. d8 cis8 fis8 b,8 e16 d16 
  cis2. 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  gis8 
    fis4 cis4 d4 dis4 
    e4 b4 e,4 e'4 
    a4 b4 cis4 b8 a8 
    b4 b,4 e4 } cis4 
  fis8 gis8 a4 e4 cis8 a8   %5
  d4 cis8 d8 e4 cis4 
  b4 cis4 d8 g8 fis4 
  b,2. cis4 
  fis4 e8 d8 cis4 b4 
  a8 b8 cis8 d8 e4 b4   %10
  fis'8 gis8 a4 gis8 e8 a4~ 
  a8 gis8 fis4 e8 d8 e4 
  a,2. 
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