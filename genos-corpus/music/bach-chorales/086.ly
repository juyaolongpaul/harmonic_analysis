\version "2.10.33"

\header {
  title = "86 - Wie schön leuchtet der Morgenstern"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d4 
    a'4 fis8 e8 d4 a'4 
    b8. cis32 d32 b4 a4 a4 
    b4 cis4 d4 cis4 
    b8 d8 cis8 b8 a4 fis4 
    b4 a4 g4 fis4     %5
    e2 d4 } r4
  a'2 fis2
  a2 fis2
  fis4 fis4 e4 e4
  fis4 fis4 e4 fis4  %10
  g4 fis4 e2
  d2 d'4 cis4
  b4 a4 g4 fis4
  e2 d4 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 a4 b4 cis4 
    fis4 e4 e4 fis4 
    g4 g4 fis4 e8 a8 
    a4 gis4 e4 d4 
    d4 d4 d8 cis8 d4     %5
    d4 cis4 a4 } r4
  e'2 d2
  e2 d2
  d4 d4 cis4 cis4
  d4 d4 cis4 dis4   %10
  e4 d8 cis8 b4 cis4
  a2 fis'4 fis8 e8
  d8 e8 fis4 fis8 e8 e8 d8
  d8 cis16 b16 cis4 a4
}

tenor = \relative c {
  \repeat volta 2 {
    \partial 4 fis4 
    e4 d8 e8 fis8 gis8 a4 
    a4 gis4 cis4 d4 
    d4 e4 a,8 b8 cis4 
    fis,8 fis'8 e8 d8 cis4 a4 
    g4 a4 b8 g8 a4     %5
    a4. g8 fis4 } r4
  a2 a2
  cis2 a2
  a4 a4 a4 a4
  a4 a4 a4 a4  %10
  b8 cis8 d4 d8 cis16 b16 a8 g8
  fis2 fis8 gis8 ais4
  b4 cis4 d8 e8 a,4
  b4 a8 g8 fis4 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 d4 
    cis4 d8 cis8 b4 fis'8 e8 
    d8 b8 e4 a,4 d4 
    g4 fis8 e8 fis8 gis8 a4 
    d,4 e4 a,4 d4 
    g4 fis4 e4 d4     %5
    a2 d4 } r4
  cis2 d2
  a2 d2
  d8 e8 fis8 g8 a8 b8 a8 g8
  fis8 e8 fis8 d8 a8 a'8 g8 fis8  %10
  e4 b'8 a8 g4 a4
  d,2 b4 fis4
  g4 a4 b8 cis8 d4
  g,4 a4 d,4 
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