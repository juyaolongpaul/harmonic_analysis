\version "2.10.33"

\header {
  title = "32 - Nun danket alle Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 e4 
    e4 e4 fis4 fis4 
    e2 r4 cis4 
    d4 cis4 b4 cis8. d16 
    b2 a4 } b4 
  b4 b4 cis4 cis4   %5
  b2 r4 b4 
  cis8 dis8 e4 e4 dis4 
  e2 r4 e4 
  fis4 e4 d4 cis4 
  d2 r4 cis4   %10
  b4 cis8. d16 b4. a8 
  a2. 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 a4 a4 a4 
    a2 r4 a4 
    b4 a4 gis4 a4 
    a8 fis8 gis4 e4 } gis8 fis8 
  e4 e4 e4 e4   %5
  e2 r4 gis4 
  a4 gis4 fis4 fis4 
  gis2 r4 a4 
  a8 b8 cis4 cis8 b4 ais8 
  b2 r4 a4   %10
  gis4 a4 a4 gis4 
  e2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 cis4 
    cis4 cis4 d4 d4 
    cis2 r4 fis4 
    e4 e4 e4 e4 
    e8 d16 cis16 d4 cis4 } b8 a8 
  gis8 a8 b4 b8 a16 gis16 a4   %5
  gis2 r4 e'4 
  e4 e4 cis4 b4 
  b2 r4 cis4 
  d4 cis4 fis4 g8 fis8 
  fis2 r4 e4   %10
  e4 e8 fis8 fis8 d8 b8. e16 
  cis2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    a'8 b8 cis8 a8 d,8 e8 fis8 d8 
    a'2 r4 a4 
    gis4 a4 e8 d8 cis8 a8 
    e'2 a,4 } e'4 
  e8 fis8 gis8 e8 a,8 b8 cis8 a8   %5
  e2 r4 e'4 
  a8 b8 cis8 gis8 a8 fis8 b8 b,8 
  e2 r4 a,4 
  d8 cis8 b8 ais8 b8 d8 e8 fis8 
  b,2 r4 cis8 d8   %10
  e4 a8 fis8 d8 b8 e8 e,8 
  a2. 
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