\version "2.10.33"

\header {
  title = "330 - Nun danket alle Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 d4 
    d4 d4 e4 e4 
    d2. b4 
    c4 b4 a4 b8 c8 
    a2 g4 } a4 
  a4 a4 b4 b4   %5
  a2. a4 
  b8 cis8 d4 d4 cis4 
  d2. d4 
  e4 d4 c4 b4 
  c2. b4   %10
  a4 b8 c8 a4. g8 
  g2. 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    a4 g4 g4 g8 a8 
    b2. g4 
    a4 d,4 d4 d4 
    e16 fis16 g4 fis8 d4 } fis8 g8 
  a8 g8 a8 fis8 d4 g4   %5
  fis2. fis4 
  d8 e8 fis4 b4 a8 g8 
  fis2. g4 
  g8 a8 b4 b8 a a gis8 
  a2. d,4   %10
  d4. e8 d8 g8 fis4 
  d2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    a8 d8 b4 c4 c4 
    d2. d4 
    d4. g8 fis4 g8 g,8 
    e'4 a,4 b4 } a4 
  d4 c4 b8 c8 d4   %5
  d2. d8 c8 
  b8 a8 a8 d8 d8 g8 e16 d16 e8 
  d2. b4 
  c4 d8 e16 f16 g8 c,8 f8 e8 
  e2. g4   %10
  fis8 d8 b8 a8 a8 b8 c8 a8 
  b2. 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    fis4 g4 c,8 d8 e8 c8 
    g'2. g4 
    g8 fis8 g4 d8 c8 b8 e8 
    c8 b16 c16 d4 g,4 } d'8 e8 
  fis8 e8 fis8 d8 g,8 a8 b8 c8   %5
  d2. d4 
  g4 fis8 b8 g8 e8 a8 a,8 
  d2. g4 
  c,4 g'8 f8 e8 f8 d8 e8 
  a,2. b8 g8   %10
  d'8 fis8 g8 c,8 d4 d,4 
  g2. 
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