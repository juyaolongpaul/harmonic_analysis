\version "2.10.33"

\header {
  title = "114 - Von Gott will ich nicht lassen"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 a4 
    a4 b4 c4 d4 
    b4. a8 g4 g4 
    a4 a4 b4 b4 
    e,2. } e'4 
  d4 c4 b4 a4   %5
  b2. b4 
  c4 d4 e4 e4 
  d2 b4 d4 
  c4 b4 a4 a4 
  b4. a8 g4 f4   %10
  e4 a8 b8 c4 b4 
  a1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    f4 f4 g4 a4 
    g4 fis4 e4 e4 
    e4 d8 e8 f4 e8 d8~ 
    d8 c16 b16 c2 } e8 fis8 
  gis4 a8 g8 f4 e8 fis8   %5
  gis2. gis4 
  a4 b4 c8 b8 a8 g8 
  a8 g8 fis4 g4 f4 
  e8 a4 g8 g8 f16 e16 f4 
  fis8 dis8 e8 fis8 b,4 b4   %10
  c4 c8 d8 e4 e8 d8 
  c1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c4 
    c4 d4 g,4 d'4 
    d8 e8 fis4 b,4 b4 
    a4 a4 a4 gis4 
    a2. } g8 a8 
  b4 e8 c8 d8 b8 c4   %5
  b2. e4 
  e4 f4 g4 c,8 b8 
  a4 d4 d4 b4 
  a8 e'8 e4 e8 d16 cis16 d8 c8 
  b8 a8 g8 fis8 e4 f4   %10
  g4 a4 a4 gis4 
  a1 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  g8 
    f4 e8 d8 e4 fis4 
    g4 dis4 e4 e8 d8 
    c4 f8 e8 d4 e4 
    a2. } c,4 
  b8 b'8 a4 gis4 a4   %5
  e2. e4 
  a4 g8 f8 c'4 e,4 
  fis4 e8 d8 g4 gis4 
  a4 e4 f4. e8 
  dis8 b8 cis8 dis8 e4 d4   %10
  c4 f4 e8 d8 e4 
  a,1 
  
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