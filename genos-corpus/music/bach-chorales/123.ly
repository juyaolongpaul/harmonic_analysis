\version "2.10.33"

\header {
  title = "123 - Helft mir Gotts Güte preisen"
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
    b8 c4 b16 c16 d4 e4 
    d8 c8 b8 a8 c4 b8 a8 
    a2. } e'4 
  d4 c4 b4 a4   %5
  b2. b4 
  c4 d4 e4 e4 
  d8 c8 b8 c8 b4 b4 
  c4 d4 e4 e4 
  d8 c8 b8 c8 b4 e4   %10
  d8 c8 b8 a8 c4 b8 a8 
  a1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e8  d8 
    e8 f8 e8 d8 c8 e8 a8 fis8 
    d8 g8 e8 a8 gis4 b8 gis8 
    e4. a8 a8 b8 gis4 
    e2. } a8 g8 
  fis8 gis8 a4 a8 gis8 a8 d8   %5
  gis,2. gis4 
  a4 g4 g4 c8 b8 
  a8 g8 fis4 g4 b4 
  a4 g4 g4 g4 
  fis8 gis8 a4 gis4 c8 b8   %10
  a4 gis8 a8 a8 fis8 gis4 
  e1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 c8  b8 
    c8 d8 c8 b8 a8 c8 b8 a8 
    b8 e8 c8 f8 e8 d8 b4 
    a8 b8 c8 d8 e8 f8 b,8 d8 
    c2. } c4 
  c8 b8 e8 f8 b,8 e8 e8 a,8   %5
  e'2. e4 
  e4 d4 c8 d8 e4 
  a,4 d4 d4 e4 
  e4 b4 c4 b4 
  a8 e'8 d8 a8 e'4 a,4   %10
  d8 e8 f8 d8 e4 d8 cis8 
  cis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 a4 
    a'4 gis4 a4 fis4 
    g8 a4 g16 a16 b4 gis4 
    a4 f4 e8 d8 e4 
    a,2. } a4 
  b4 c8 d8 e4 f4   %5
  e2. e4 
  a8 b8 c8 b8 c8 b8 a8 g8 
  fis8 e8 d4 g4 gis4 
  a8 g8 f8 g8 f8 e8 d8 c8 
  d8 e8 f8 d8 e4 c4   %10
  f8 e8 d8 f8 e8 d8 e4 
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