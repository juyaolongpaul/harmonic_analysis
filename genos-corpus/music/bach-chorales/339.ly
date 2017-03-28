\version "2.10.33"

\header {
  title = "339 - Wer nur den lieben Gott läßt walten"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    a4 b4 c4 b4 
    a4 b4 gis4. fis8 
    e4 g4 g4 f4 
    e4 a4 a4 gis4 
    a2. } b4   %5
  c4 d4 e4 e4 
  d4 d4 c4 e4 
  d4 c4 b4 a4 
  c4 b4 a4 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e16 d16 f4 e16 d16 c8 a'4 g8~ 
    g8 fis16 e16 fis8 fis8 fis8 e8 d8 c8 
    b4 e8 d8 cis4 d4 
    b8 c8 d8 e8 fis4 e4 
    e2. } gis4   %5
  a4 b4 c8 g8 c8 bes8 
  a4 b4 g4 g4 
  f4 fis4 gis4 a4 
  dis,8 fis8 f8 e16 d16 cis4 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c8 b16 a16 gis8 e'8 e4 dis4 
    e4 b4 b4. a8 
    gis4 e4 a4 a4 
    gis8 a8 b8 c8 c8 b8 b4 
    cis2. } e4   %5
  e16 c16 f4 e16 d16 c8 d8 e4 
  f8 d8 g8 f8 e4 c4 
  c8 b8 a8 gis16 a16 e8 d'8 c8 b8 
  a4. gis8 e4 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  g8 
    f4 e4 a,4 b4 
    cis4 dis4 e2 
    e4 cis4 a4 d4 
    d8 c8 f8 e8 dis4 e4 
    a,2. } e'4   %5
  a4 g8 f8 e8 d8 c4 
  d8 e16 f16 g8 g,8 c4 c4 
  d4 dis4 e4 f4 
  fis8 dis8 e8 e,8 a4 
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