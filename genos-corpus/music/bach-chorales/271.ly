\version "2.10.33"

\header {
  title = "271 - Gib dich zufrieden und sei stille"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    \partial 4 b4 
    g4 a8 b16 c16 b4. a8 
    g4 fis8 g16 a16 g4. fis8 
    e2 c'4 b4 
    e4 d4 fis,8 c'8 b8 a8 
    a2 g4 } b4   %5
  a4 b8 g8 a4 b8 c8 
  d4 c8 b8 c8 b8 a4 
  r4 c4 fis,4 e'8 dis8 
  e8 d8 c8 b8 a8 g'8 fis8 e8 
  dis8 cis8 b4 r4 e4   %10
  c4 b8 a8 b8 dis,8 e8 a8 
  g4 fis8 e8 e4
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    e4 fis8 g16 a16 g8 dis8 e8 fis8 
    fis8 e8 e8 dis8 e4. dis8 
    e2 g4. fis8 
    e8 fis8 fis8 g8 fis4 g4 
    g4 fis4 d4 } g4   %5
  a8 fis8 g8 e8 fis4 e4 
  d4 e4 e8 d8 e4 
  r4 e4 dis8 e8 fis4 
  e8 fis8 g8 fis8 e4 a8 b16 c16 
  fis,8 e8 dis4 r4 e4   %10
  e4. fis8 dis8 fis8 e8 e8 
  e4 dis b
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    b4 c8 fis,8 g8 a8 b8 c8 
    b4 b4 b8 g8 a4 
    g2 g8 a8 b4 
    b8 a8 a8 b8 c4 d4 
    e4 d8 c8 b4 } d4   %5
  d4 d8 cis8 d4 gis,8 a8 
  a8 b8 gis4 a8 b8 c4 
  r4 c4 b4. c8 
  b8 a8 g8 gis8 a8 b8 c4 
  b4 fis4 r4 b4   %10
  a8 b8 c4 fis,8 a8 b8 a8 
  b4 a8 g8 g4
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4 
    e8 dis8 e8 dis8 e8 fis8 g8 a8 
    b4 b,4 e2 
    e2 e4. d8 
    c8 c'4 b8 a4 g4 
    c,4 d4 g,4 } g'4   %5
  fis8 d8 g4 d8 e8 d8 c8 
  b4 e4 a4 a,4 
  r4 a'4 b8 c8 b8 a8 
  g8 fis8 e8 d8 c8 b8 a4 
  b4 b4 r4 gis'4   %10
  a8 a,8 a'4. fis8 g8 c,8 
  b8 a8 b4 e 
  
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