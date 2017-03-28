\version "2.10.33"

\header {
  title = "161 - Ihr Gestirn', ihr hohlen Lüfte"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    d2 e4 
    f2 g4 
    a2 d4 
    cis2 cis4 
    d2 e4     %5
    f2. 
    f,4 g4. g8 
    f2. }
  g2 g4 
  a2 c4   %10
  bes2 g4 
  a2 a4 
  c2 c4 
  d2 f4 
  e4 d4 cis4   %15
  d2 d4
}

alto = \relative c' {
  \repeat volta 2 {
    a2 cis4 
    d2 e4 
    f4. g8 a4 
    a2 a4 
    a2 a4     %5
    a4. g8 f8 e8 
    f2 e4 
    c2. }
  e4 f4 g4~ 
  g8 f4 e8 f4   %10
  f4 e8 d8 e4 
  f2 f4 
  a2 a8 g8 
  f8 e8 f8 g8 a4 
  bes8 g8 f8 e8 e8 g8   %15
  fis2 fis4 
}

tenor = \relative c {
  \repeat volta 2 {
    f8 g8 a4 g4 
    a2 bes4 
    c4 d8 e8 f4 
    e2 e4 
    f2 e4     %5
    d8 cis8 d8 e8 d8 c8 
    bes8 a8 d4 c8. bes16 
    a2. }
  c2 c4 
  c4. bes8 a4   %10
  d2 c4 
  c2 c4 
  f2 f8 ees8 
  d2 c4 
  bes4 b4 a4   %15
  a2 a4
}

baixo = \relative c {
  \repeat volta 2 {
    d8 e8 f8 g8 f8 e8 
    d4 c8 bes8 a8 g8 
    f8 g8 f8 e8 d4 
    a'2 a4 
    d8 e8 f8 e8 d8 cis8     %5
    d2.~ 
    d8 c8 bes4 c4 }
  f,2. 
  c'4 d4 e4 
  f4 g4 a4   %10
  g4 g,4 c4 
  f2 f4 
  f4 f,8 g8 a4 
  bes4 bes'4 a4 
  g4 gis4 a4   %15
  d,2 d4
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