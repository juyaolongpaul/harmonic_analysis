\version "2.10.33"

\header {
  title = "268 - Nun lob', mein' Seel', den Herren"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 b4 a4 g4 
    c4 d4 e4 e4 
    e8 d8 e4 e4 d4 
    c8 d8 d4 c4 } c4 
  c8 d8 e4 d4 e4   %5
  c4 b4 a4 a4 
  d8 c8 b4 c8 b8 a4 
  g4 g4 c4 c4 
  d4 d4 e8 d8 e4 
  c4 c4 f4 f4   %10
  e8 d8 e4 d4 d4 
  e4 e4 f4 f4 
  g8 f16 e16 f8 g8 c,4 e4 
  d8 c8 b4 c8 b8 a4 
  g4 g4 c4 b4   %15
  a4 g4 d'4 e4 
  d4 e4 f4 e4 
  d4 c8 d8 e4 d4 
  c2. 
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    a4 g4 g8 f8 e8 f8 
    g4 a8 g8 g4 g4 
    g8 a8 b4 a4. g8 
    g8 a8 g8 f8 e4 } e4 
  f4 c'4 g8 f8 e4   %5
  e2 e4 e4 
  d4 d4 g8 e8 fis4 
  d4 d4 g8 f8 e4 
  a8 f8 g4 g8 f8 g4 
  f4 g4 f8 g8 a4   %10
  bes4 e,4 f4 g4 
  g4 a4 a4 bes4 
  bes4 a8 g8 f4 g4 
  a4 g4 g8 e8 fis4 
  d4 d4 a'4 d,8 e8   %15
  f4 g4 a4. b16 c16 
  g4 b4 a4 g8 a8 
  b4 c2 b4 
  g2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e4 e8 d8 c4 c4 
    c4. b8 c4 c8 d8 
    e8 f8 e8 d8 c4. b8 
    c4 b4 g4 } g4 
  a4 g8 a8 b4 b4   %5
  b8 a4 gis8 c4 c8 b8 
  a4 g4 g4 d'8 c8 
  b4 b8 a8 g4 c4 
  c4 b4 c4 bes4 
  a4 c4 d8 e8 f4   %10
  g4 cis,4 a4 b4 
  c4 c4 d4 d4 
  e8 d8 c8 bes8 a4 c8 b8 
  a4 d4 g,4 d'8 c8 
  b4 b4 e,8 fis8 g4   %15
  d'4 d8 c8 c8 b8 c4 
  b4 e8 d8 c8 d8 e8 f8 
  g8 f8 e8 f8 g4. f8 
  e2. 
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 c8  b8 
    a4 e4 f4 c8 d8 
    e4 f8 g8 c,4 c4 
    c'4 gis4 a8 g8 f8 g8 
    e8 f8 g4 c,4 } c4 
  f4 e8 f8 g4 gis4   %5
  a4 e4 a,4 a'8 g8 
  fis4 g8 fis8 e8 c8 d4 
  g,4 g'8 f8 e4 a8 g8 
  f8 d8 g4 c,4. d16 e16 
  f4 e4 d4 d'4   %10
  g,8 bes8 a4 d,4 g4 
  c8 b8 a8 g8 f8 e8 d4 
  c4 d8 e8 f4 c'4 
  fis,4 g4 e8 c8 d4 
  g,4 g4 a4 b8 c8   %15
  d4 e4 f4 fis4 
  g4 gis4 a8 b8 c4 
  f,8 g8 a4 g8 f8 g4 
  c,2. 
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