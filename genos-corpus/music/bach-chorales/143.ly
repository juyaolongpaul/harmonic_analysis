\version "2.10.33"

\header {
  title = "143 - In dulci jubilo"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key f \major
}


soprano = \relative c' {
  \partial 4 f4 
    f2 f4 
  a2 bes4 
  c2 d4 
  c2. 
  f,2 f4   %5
  a2 bes4 
  c2 d4 
  c2. 
  c2 d4 
  c2 bes4   %10
  a4. bes8 a8 g8 
  f2 f4 
  g2 g4 
  a2 g4 
  f2 g4   %15
  a2. 
  c2 d4 
  c2 bes4 
  a4. bes8 a8 g8 
  f2 f4   %20
  g2 g4 
  a2 g4 
  f2 g4 
  a2. 
  d,2 d4   %25
  e2 e4 
  f8 e8 f8 g8 a8 bes8 
  c2. 
  a2 a4 
  g2 g4   %30
  f2.~ 
  f2. 
  
}

alto = \relative c' {
  \partial 4 c4 
    d2 d4 
  e2 e4 
  f2. 
  f2. 
  f4 e4 d4   %5
  c2 d4 
  c4 e4 g4 
  e2. 
  f2 f4 
  e4 f4 g4   %10
  cis,4 d4 e4 
  d2 f4 
  f4 e8 d8 e4 
  f2 e4 
  f4. g8 f8 e8   %15
  f2. 
  f2 f4 
  g4 f4 g4 
  g4 f4 e4 
  d2 f4~   %20
  f4 e8 d8 e4 
  f2 e4~ 
  e4 a,4 d4 
  cis2. 
  d2 d4~   %25
  d4 c8 d8 e4~ 
  e4 d2 
  c2. 
  c4 f8 e8 f4 
  f2 e4   %30
  f8 ees8 d8 c8 d8 bes8 
  c2. 
  
}

tenor = \relative c' {
  \partial 4 a4 
    a2 bes4 
  c2 bes4 
  a2 bes4 
  a2. 
  a2 bes4   %5
  a4 g4 f4 
  g4 c4 b4 
  g2. 
  a2 bes4 
  c4 d4 e4   %10
  a,4 b4 cis4 
  a2 c4 
  d4 bes4 c4 
  c2 c4 
  c4 a4 c4   %15
  c2. 
  a2 bes4 
  g2 e'4 
  e4 d4 cis4 
  a2 c4   %20
  d4 bes4 c4 
  c2 c4 
  a2 bes4 
  e,2. 
  r8 f8 g8 a8 bes8 a8   %25
  g4 a8 bes8 c8 bes8 
  a4. g8 f4 
  g2. 
  a4 bes4 c4 
  d4 bes4 c4~   %30
  c4 bes8 a8 bes8 g8 
  a2. 
  
}

baixo = \relative c {
  \partial 4 f4 
    d4 c4 bes4 
  a4 bes8 a8 g8 a8 
  f8 f'8 ees8 d8 c8 bes8 
  f'2. 
  d4 c4 bes4   %5
  f'4 e4 d4 
  e4 c4 g4 
  c2. 
  f2 bes4~ 
  bes4 a4 g4~   %10
  g4 f4 e4 
  d2 a4 
  bes4 g4 c4 
  f,8 g8 a8 bes8 c8 bes8 
  a4 f4 c4   %15
  f2. 
  f'4 e4 d4 
  e4 d4 c4 
  f8 g8 a4 a,4 
  d2 a4   %20
  bes4 g4 c4 
  f,8 g8 a8 bes8 c8 a8 
  d4 c4 bes4 
  a2. 
  bes4 a4 g4   %25
  c4 bes4 a4 
  d8 c8 d8 e8 f8 d8 
  e2. 
  f,4 g4 a4 
  bes4 g4 c4   %30
  a4 bes2 
  f2. 
  
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