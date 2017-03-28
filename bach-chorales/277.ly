\version "2.10.33"

\header {
  title = "277 - Herzlich lieb hab'ich dich, o Herr"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 c4 
    b4 a4 g8 f8 e4 
    a4 a4 g4 c4 
    b4 a4 g8 f8 e4 
    a4 a4 g4 g4 
    a4 b4 c8 d8 e4     %5
    d2 c4 } c4 
  c4 d4 c4 a8 b8 
  c4. d8 b4 b4 
  c4 d4 c8 b8 a8 b8 
  c4. d8 b4 e4   %10
  f8 e8 d4 e8 d8 c4 
  c4 b4 c4 b4 
  c4 b4 a8 b8 g4 
  g8 a8 fis4 g4 e'4 
  d4 c4 b4 e4   %15
  f8 e8 d4 e8 d8 c8 b8 
  a8 d8 b4 c4 c4 
  c4 c4 d4 c4 
  c4 b4 c4 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    \partial 4 g4 
    g8 f8 e4 e4 e8 d8 
    c4 f4 e4 e8 f8 
    g4 f8 e8 d4 c8 d8 
    e8 dis8 e8 fis8 fis8 e8 e4 
    d8 c8 f4 e4 a4     %5
    a4 g8 f8 e4 } g4 
  a4 b4 a4 e4 
  e4 d4 d4 g4 
  g4 g4 g4 f4 
  e4. fis8 gis4 a8 g8   %10
  f4 g4 g8 f8 e4 
  a4 g4 g4 g4 
  a4 a8 g8 g8 fis8 fis8 e8 
  e4 d4 d4 c4 
  d8 e8 e8 dis8 e4 e8 d8   %15
  c8 e8 gis8 a8 b4 e,4 
  e8 d8 e4 e4 c'8 bes8 
  a8 g8 f8 e8 d4 e8 g8 
  g4. g8 g4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    d4 d8 c8 b4 c8 b8 
    a8 b8 c4 c4 c8 d8 
    e8 d8 c4 d4 g,4 
    c8 b8 a8 c8 b4 c8 b8 
    a4 d4 c8 b8 c4     %5
    c4 b4 g4 } e'4 
  e8 c8 f8 e8 e4. d8 
  c8 b8 a4 g4 d'4 
  e4 f4 e4 a,8 gis8 
  a8 e8 a4 b4 a4   %10
  d8 c8 b4 b4 c4 
  d4 e8 f8 e4 d4 
  d4 d8 e8 e8 d8 b8 c8 
  c4 a8 b16 c16 b4 g8 a8 
  b4 a8 fis8 gis4 c8 b8   %15
  a8 c8 d8 f8 b,4 c4 
  c8 a8 b8 gis8 a4 g4 
  c8 bes8 a4 a8 g8 g8 e'8 
  d8 c8 d4 e4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 c4 
    g'4 a4 e8 d8 c4 
    f8 g8 a8 b8 c4 a4 
    e4 f4 b,4 c8 b8 
    a8 b8 cis8 dis8 e4 c4 
    f4 e8 d8 a'4. g8     %5
    f8 d8 g4 c,4 } c'8 b8 
  a4. gis8 a8 b8 c8 b8 
  a8 g8 fis8 d8 g4 g8 f8 
  e8 d8 c8 b8 c4 d4 
  a8 b8 c8 a8 e'4 cis4   %10
  d4 g8 f8 e4 a8 g8 
  f8 d8 g4 c,4 g'4 
  g8 fis8 g8 e8 c8 d8 e8 c8 
  a4 d4 g,4 c4 
  g'8 gis8 a4 e4 a,4   %15
  a'4 b8 a8 gis4 a8 g8 
  fis4 gis8 e8 a4 e4 
  f8 g8 a4 b,4 e8 c8 
  g'4 g,4 c4 
  
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