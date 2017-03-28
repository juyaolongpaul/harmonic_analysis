\version "2.10.33"

\header {
  title = "340 - Befiehl du deine Wege"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  e8 
    f4 e4 f4 g4 
    a2 a4 c4 
    b4 a4 a4 gis4 
    a2. } a4 
  b4 c4 d8 c8 b4   %5
  c4. bes8 a4 a4 
  g4 f4 f4 e4 
  f2. a8 g8 
  f4 g4 a4 g4 
  f2 e4 f4   %10
  g4 f4 e2 
  d2. 
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 d8  cis8 
    d4 bes4 a4 d8 c8 
    c2 c4 c4 
    f4 fis4 e4 e4 
    e2. } e8 fis8 
  gis4 a4 a4 g4   %5
  g4 f8 g8 f4 f8 ees8 
  d4 d4 c4 c4 
  c2. e4 
  f4. e8 f4. e8~ 
  e8 d8 d4 cis4 d4~   %10
  d8 cis8 d2 cis4 
  a2. 
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 a8  g8 
    a4 g4 f4 f8 e8 
    f2 f4 g4 
    gis4 c4 b8 c8 d4 
    c2. } c4 
  d4 e4 d4 d4   %5
  c8 e8 d8 c8 c4 c4 
  bes4 aes4 g4. a16 bes16 
  a2. a4 
  a4 d8 c8 c4 d8 e8 
  a,8 gis8 a8 b8 a4 a4   %10
  g4 a4 a4. g8 
  f2. 
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 f8  e8 
    d4. cis8 d8 c8 bes8 c8 
    f,2 f4 e'4 
    d4 dis4 e4 e,4 
    a2. } a'4~ 
  a4 a8 g8 fis4 g8 f8   %5
  e8 c8 d8 e8 f4 f,4 
  bes4 b4 c4 c,4 
  f2. cis'4 
  d8 c8 bes8 c8 f,8 a8 b8 cis8 
  d8 bes8 a8 gis8 a4 d4   %10
  e4 f8 g8 a4 a,4 
  d2. 
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